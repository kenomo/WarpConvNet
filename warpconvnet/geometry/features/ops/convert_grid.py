# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Operations for converting grid features between different memory formats.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


def convert_to_standard_format(
    tensor: Tensor,
    from_format: "GridMemoryFormat",  # noqa: F821
    num_channels: int,
    grid_shape: Tuple[int, int, int],
) -> Tensor:
    """Convert tensor from any memory format to standard b_x_y_z_c format.

    Args:
        tensor: Input tensor in specified memory format
        from_format: Current memory format of the tensor
        num_channels: Number of feature channels
        grid_shape: 3D grid shape (H, W, D)

    Returns:
        Tensor in b_x_y_z_c format
    """
    from ..grid import GridMemoryFormat

    H, W, D = grid_shape

    if from_format == GridMemoryFormat.b_x_y_z_c:
        return tensor

    if from_format == GridMemoryFormat.b_c_x_y_z:
        return tensor.permute(0, 2, 3, 4, 1)

    if from_format == GridMemoryFormat.b_zc_x_y:
        B, ZC, H_tensor, W_tensor = tensor.shape
        assert ZC == D * num_channels, f"Expected Z*C={D*num_channels}, got {ZC}"
        assert (
            H_tensor == H and W_tensor == W
        ), f"Expected shape ({H}, {W}), got ({H_tensor}, {W_tensor})"
        return tensor.reshape(B, D, num_channels, H, W).permute(0, 3, 4, 1, 2)

    if from_format == GridMemoryFormat.b_xc_y_z:
        B, XC, W_tensor, D_tensor = tensor.shape
        assert XC == H * num_channels, f"Expected X*C={H*num_channels}, got {XC}"
        assert (
            W_tensor == W and D_tensor == D
        ), f"Expected shape ({W}, {D}), got ({W_tensor}, {D_tensor})"
        return tensor.reshape(B, H, num_channels, W, D).permute(0, 1, 3, 4, 2)

    if from_format == GridMemoryFormat.b_yc_x_z:
        B, YC, H_tensor, D_tensor = tensor.shape
        assert YC == W * num_channels, f"Expected Y*C={W*num_channels}, got {YC}"
        assert (
            H_tensor == H and D_tensor == D
        ), f"Expected shape ({H}, {D}), got ({H_tensor}, {D_tensor})"
        return tensor.reshape(B, W, num_channels, H, D).permute(0, 3, 1, 4, 2)

    raise ValueError(f"Unsupported memory format: {from_format}")


def convert_from_standard_format(
    tensor: Tensor,
    to_format: "GridMemoryFormat",  # noqa: F821
    grid_shape: Optional[Tuple[int, int, int]] = None,
) -> Tensor:
    """Convert tensor from standard b_x_y_z_c format to specified format.

    Args:
        tensor: Input tensor in b_x_y_z_c format
        to_format: Target memory format
        grid_shape: Optional explicit grid shape for validation

    Returns:
        Tensor in requested format
    """
    from ..grid import GridMemoryFormat

    B, H, W, D, C = tensor.shape

    if grid_shape is not None:
        H_req, W_req, D_req = grid_shape
        assert (
            H == H_req and W == W_req and D == D_req
        ), f"Expected shape {grid_shape}, got ({H}, {W}, {D})"

    if to_format == GridMemoryFormat.b_x_y_z_c:
        return tensor

    if to_format == GridMemoryFormat.b_c_x_y_z:
        return tensor.permute(0, 4, 1, 2, 3)

    if to_format == GridMemoryFormat.b_zc_x_y:
        return tensor.permute(0, 3, 4, 1, 2).reshape(B, D * C, H, W)

    if to_format == GridMemoryFormat.b_xc_y_z:
        return tensor.permute(0, 1, 4, 2, 3).reshape(B, H * C, W, D)

    if to_format == GridMemoryFormat.b_yc_x_z:
        return tensor.permute(0, 2, 4, 1, 3).reshape(B, W * C, H, D)

    raise ValueError(f"Unsupported memory format: {to_format}")
