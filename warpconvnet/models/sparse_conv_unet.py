from typing import List, Optional

import torch
import torch.nn as nn
import warp as wp
from jaxtyping import Float
from torch import Tensor

from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.nn.activations import ReLU
from warpconvnet.nn.functional.transforms import apply_feature_transform
from warpconvnet.nn.normalizations import LayerNorm
from warpconvnet.nn.sparse_conv import SPATIALLY_SPARSE_CONV_ALGO_MODE, SparseConv3d


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        channel_multiplier: int = 2,
        kernel_search_batch_size: int = 8,
        kernel_matmul_batch_size: int = 2,
        conv_algo: SPATIALLY_SPARSE_CONV_ALGO_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM,
    ):
        super().__init__()
        intermediate_dim = max(in_channels, out_channels) * channel_multiplier
        self.conv1 = SparseConv3d(
            in_channels,
            intermediate_dim,
            kernel_size=kernel_size,
            kernel_search_batch_size=kernel_search_batch_size,
            kernel_matmul_batch_size=kernel_matmul_batch_size,
            conv_algo=conv_algo,
        )
        self.norm1 = LayerNorm(intermediate_dim)
        self.conv2 = SparseConv3d(
            intermediate_dim,
            out_channels,
            kernel_size=kernel_size,
            kernel_search_batch_size=kernel_search_batch_size,
            kernel_matmul_batch_size=kernel_matmul_batch_size,
            conv_algo=conv_algo,
        )
        self.norm2 = LayerNorm(out_channels)
        self.relu = ReLU()
        if in_channels != out_channels:
            self.identity = nn.Linear(in_channels, out_channels)
        else:
            self.identity = nn.Identity()

    def forward(self, x: SpatiallySparseTensor) -> SpatiallySparseTensor:
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        identity = apply_feature_transform(identity, self.identity)
        x = x + identity
        x = self.relu(x)
        return x


class SparseConvEncoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        num_blocks_per_level: List[int] | int,
        kernel_sizes: List[int] | int,
        channel_multiplier: int = 2,
        num_levels: int = 4,
        kernel_search_batch_size: int = 8,
        kernel_matmul_batch_size: int = 2,
        conv_algo: SPATIALLY_SPARSE_CONV_ALGO_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM,
    ):
        super().__init__()
        self.num_levels = num_levels
        if isinstance(num_blocks_per_level, int):
            num_blocks_per_level = [num_blocks_per_level] * num_levels
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_levels
        assert len(encoder_channels) == num_levels + 1
        assert len(num_blocks_per_level) == self.num_levels
        assert len(kernel_sizes) == self.num_levels

        self.down_blocks = nn.ModuleList()
        self.level_blocks = nn.ModuleList()
        for level in range(self.num_levels):
            in_channels = encoder_channels[level]
            out_channels = encoder_channels[level + 1]

            down_block = nn.Sequential(
                SparseConv3d(
                    in_channels,
                    out_channels,
                    stride=2,
                    kernel_size=kernel_sizes[level],
                    kernel_search_batch_size=kernel_search_batch_size,
                    kernel_matmul_batch_size=kernel_matmul_batch_size,
                    conv_algo=conv_algo,
                ),
                LayerNorm(out_channels),
                ReLU(),
            )
            self.down_blocks.append(down_block)

            level_block = [
                ResBlock(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_sizes[level],
                    channel_multiplier=channel_multiplier,
                )
            ]
            for _ in range(num_blocks_per_level[level]):
                level_block.append(
                    ResBlock(
                        out_channels,
                        out_channels,
                        kernel_size=kernel_sizes[level],
                        channel_multiplier=channel_multiplier,
                    )
                )
            self.level_blocks.append(nn.Sequential(*level_block))

    def forward(self, x: SpatiallySparseTensor) -> List[SpatiallySparseTensor]:
        out_features = [x]
        for level in range(self.num_levels):
            x = self.down_blocks[level](x)
            x = self.level_blocks[level](x)
            out_features.append(x)
        return out_features


class SparseConvDecoder(nn.Module):
    def __init__(
        self,
        decoder_channels: List[int],
        encoder_channels: List[int],
        num_blocks_per_level: List[int] | int,
        kernel_sizes: List[int] | int,
        channel_multiplier: int = 2,
        num_levels: int = 4,
        kernel_search_batch_size: int = 8,
        kernel_matmul_batch_size: int = 2,
        conv_algo: SPATIALLY_SPARSE_CONV_ALGO_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM,
    ):
        super().__init__()
        self.num_levels = num_levels
        if isinstance(num_blocks_per_level, int):
            num_blocks_per_level = [num_blocks_per_level] * num_levels
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_levels
        assert len(decoder_channels) == num_levels + 1
        assert len(num_blocks_per_level) == self.num_levels
        assert len(kernel_sizes) == self.num_levels
        assert len(encoder_channels) >= num_levels + 1
        assert encoder_channels[-1] == decoder_channels[0]

        self.up_convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.level_blocks = nn.ModuleList()

        for level in range(self.num_levels):
            in_channels = decoder_channels[level]
            out_channels = decoder_channels[level + 1]
            enc_channels = encoder_channels[-(level + 2)]

            up_conv = SparseConv3d(
                in_channels,
                out_channels,
                transposed=True,
                kernel_size=kernel_sizes[level],
                kernel_search_batch_size=kernel_search_batch_size,
                kernel_matmul_batch_size=kernel_matmul_batch_size,
                conv_algo=conv_algo,
            )
            self.up_convs.append(up_conv)

            if enc_channels != out_channels:
                self.skips.append(
                    nn.Linear(
                        in_features=enc_channels,
                        out_features=out_channels,
                        bias=True,
                    )
                )
            else:
                self.skips.append(nn.Identity())

            level_block = [
                ResBlock(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_sizes[level],
                    channel_multiplier=channel_multiplier,
                )
            ]
            for _ in range(num_blocks_per_level[level]):
                level_block.append(
                    ResBlock(
                        out_channels,
                        out_channels,
                        kernel_size=kernel_sizes[level],
                        channel_multiplier=channel_multiplier,
                    )
                )
            self.level_blocks.append(nn.Sequential(*level_block))

    def forward(self, encoder_outputs: List[SpatiallySparseTensor]) -> List[SpatiallySparseTensor]:
        out_features = []
        x = encoder_outputs[-1]
        for level in range(self.num_levels):
            x = self.up_convs[level](x, encoder_outputs[-(level + 2)])
            x = x + apply_feature_transform(encoder_outputs[-(level + 2)], self.skips[level])
            x = self.level_blocks[level](x)
            out_features.append(x)
        return out_features
