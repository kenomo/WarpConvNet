# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from warpconvnet.geometry.types.grid import Grid, GridMemoryFormat
from warpconvnet.nn.functional.grid_conv import grid_conv
from warpconvnet.nn.modules.grid_conv import GridConv


@pytest.fixture
def create_3d_grid():
    def _create_grid(batch_size=2, channels=3, height=5, width=6, depth=4, device=None):
        # Create a grid using the standard factory method
        return Grid.from_shape(
            grid_shape=(height, width, depth),  # Grid shape is (H, W, D)
            num_channels=channels,
            memory_format=GridMemoryFormat.b_x_y_z_c,
            batch_size=batch_size,
            device=device,
        )

    return _create_grid


def test_basic_grid_conv(create_3d_grid):
    # Create input grid
    grid = create_3d_grid(batch_size=2, channels=3, height=5, width=6, depth=4)

    # Create convolution weight
    weight = torch.randn(5, 3, 3, 3, 3)  # out_channels, in_channels, D, H, W

    # Apply convolution
    output_grid = grid_conv(grid, weight, stride=1, padding=1, dilation=1, bias=None)

    # Check output dimensions (with padding=1 should preserve spatial dimensions)
    assert output_grid.grid_features.batched_tensor.shape[0] == 2  # batch size
    assert output_grid.grid_features.num_channels == 5  # output channels
    assert output_grid.grid_shape == (5, 6, 4)  # spatial dimensions
    assert output_grid.memory_format == GridMemoryFormat.b_c_z_x_y


def test_grid_conv_with_bias(create_3d_grid):
    grid = create_3d_grid(batch_size=2, channels=3, height=5, width=6, depth=4)

    weight = torch.randn(5, 3, 3, 3, 3)
    bias = torch.randn(5)

    output_grid = grid_conv(grid, weight, stride=1, padding=1, dilation=1, bias=bias)

    assert output_grid.grid_features.batched_tensor.shape[0] == 2  # batch size
    assert output_grid.grid_features.num_channels == 5  # output channels
    assert output_grid.grid_shape == (5, 6, 4)  # spatial dimensions


@pytest.mark.parametrize("stride", [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1)])
@pytest.mark.parametrize("grid_size", [(7, 8, 9), (8, 8, 8), (9, 7, 8)])
@pytest.mark.parametrize("kernel_size", [(3, 3, 3), (4, 4, 4), (5, 5, 5)])
@pytest.mark.parametrize("padding", [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
def test_grid_conv_stride(create_3d_grid, stride, grid_size, kernel_size, padding):
    D_in, H_in, W_in = grid_size
    grid = create_3d_grid(batch_size=2, channels=3, depth=D_in, height=H_in, width=W_in)
    dilation = (1, 1, 1)

    weight = torch.randn(5, 3, *kernel_size)

    output_grid = grid_conv(
        grid, weight, stride=stride, padding=padding, dilation=dilation, bias=None
    )

    # Formula defined by torch.nn.functional.conv3d
    D_out = ((D_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
    H_out = ((H_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1
    W_out = ((W_in + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2]) + 1

    # With stride=2, the spatial dimensions should be halved
    assert output_grid.grid_shape == (
        H_out,
        W_out,
        D_out,
    )  # spatial dimensions always in (H, W, D) order
    assert output_grid.grid_features.num_channels == 5  # output channels


def test_grid_conv_module(create_3d_grid):
    grid = create_3d_grid(batch_size=2, channels=3, depth=4, height=5, width=6)

    # Create the module
    conv = GridConv(
        in_channels=3,
        out_channels=5,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    )

    # Apply convolution using the module
    output_grid = conv(grid)

    # Check output dimensions (with padding=1 should preserve spatial dimensions)
    assert output_grid.grid_shape == (5, 6, 4)  # spatial dimensions
    assert output_grid.grid_features.num_channels == 5  # output channels


def test_no_bias(create_3d_grid):
    grid = create_3d_grid(batch_size=2, channels=3, depth=4, height=5, width=6)

    # Create the module without bias
    conv = GridConv(
        in_channels=3,
        out_channels=5,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    )

    # Check that bias is None
    assert conv.bias is None

    # Apply convolution
    output_grid = conv(grid)

    # Check output dimensions
    assert output_grid.grid_shape == (5, 6, 4)  # spatial dimensions
    assert output_grid.grid_features.num_channels == 5  # output channels


def test_repr():
    conv = GridConv(
        in_channels=3,
        out_channels=5,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    )

    # Check string representation
    repr_str = str(conv)
    assert "GridConv" in repr_str
    assert "in_channels=3" in repr_str
    assert "out_channels=5" in repr_str
    assert "kernel_size=(3, 3, 3)" in repr_str
