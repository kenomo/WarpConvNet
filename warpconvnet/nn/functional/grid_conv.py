# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Union, Optional, Literal
import warnings
from jaxtyping import Float

import torch
import torch.nn.functional as F
from torch import Tensor

from warpconvnet.geometry.types.grid import Grid, GridMemoryFormat
from warpconvnet.geometry.coords.grid import GridCoords


def grid_conv(
    grid: Grid,
    weight: Float[Tensor, "C_out C_in D H W"],  # noqa: F821
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    bias: bool = True,
) -> Grid:
    """
    3D Convolution on a Grid geometry type.

    It is a simple wrapper on torch.nn.functional.conv3d.
    The output grid shape is computed as follows:

    D_out = ((D_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
    H_out = ((H_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1
    W_out = ((W_in + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2]) + 1

    Args:
        grid: Grid geometry type
        weight: Weight tensor
        stride: Stride
        padding: Padding
        dilation: Dilation
        bias: Bias

    Returns:
        Grid: Output grid

    """
    # Use F.conv3d
    if grid.memory_format != GridMemoryFormat.b_c_z_x_y:
        warnings.warn(
            f"Input grid memory format is {grid.memory_format}, converting to {GridMemoryFormat.b_c_z_x_y}"
        )
        grid = grid.to_memory_format(GridMemoryFormat.b_c_z_x_y)

    # Apply convolution
    output_tensor = F.conv3d(grid.features, weight, bias, stride, padding, dilation)
    # For stride, padding, dilation, the grid shape may not match the output shape
    D, H, W = tuple(output_tensor.shape[2:])

    # Create a new Grid with the same coordinates but updated features
    return Grid(
        batched_coordinates=GridCoords.from_shape(
            grid_shape=(H, W, D),
            bounds=grid.bounds,
            batch_size=grid.batch_size,
            device=grid.device,
        ),
        batched_features=output_tensor,
        memory_format=GridMemoryFormat.b_c_z_x_y,
    )
