# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from jaxtyping import Float, Int

import torch
from torch import Tensor
import torch.nn.functional as F


def create_grid_coordinates(
    grid_shape: Tuple[int, int, int],
    bounds: Optional[Tuple[Float[Tensor, "3"], Float[Tensor, "3"]]] = None,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    flatten: bool = True,
) -> Tuple[Float[Tensor, "N 3"], Int[Tensor, "B+1"]]:  # noqa: F821
    """Create coordinate tensor for a regular grid.

    Args:
        grid_shape: Grid resolution (H, W, D)
        bounds: Min (xyz) and max (xyz) bounds (default: unit cube)
        batch_size: Number of batches
        device: Device to create tensors on
        flatten: Whether to flatten the coordinates

    Returns:
        coords: Float[Tensor, "N 3"]  # noqa: F821
        offsets: Int[Tensor, "B+1"]  # noqa: F821
    """
    H, W, D = grid_shape

    if bounds is None:
        min_bound = torch.zeros(3, device=device)
        max_bound = torch.ones(3, device=device)
    else:
        min_bound, max_bound = bounds
        min_bound = min_bound.to(device)
        max_bound = max_bound.to(device)

    # Create regular grid in the range [0, 1]
    h_coords = torch.linspace(0, 1, H, device=device)
    w_coords = torch.linspace(0, 1, W, device=device)
    d_coords = torch.linspace(0, 1, D, device=device)

    # Create meshgrid
    grid_h, grid_w, grid_d = torch.meshgrid(h_coords, w_coords, d_coords, indexing="ij")

    # Scale to bounds
    grid_h = min_bound[0] + grid_h * (max_bound[0] - min_bound[0])
    grid_w = min_bound[1] + grid_w * (max_bound[1] - min_bound[1])
    grid_d = min_bound[2] + grid_d * (max_bound[2] - min_bound[2])

    # Stack coordinates
    coords = torch.stack([grid_h, grid_w, grid_d], dim=-1)

    # Create batched coords
    if batch_size > 1:
        ndim = coords.ndim
        # Expand do not copy memory
        coords = coords.unsqueeze(0).expand(tuple([batch_size] + [-1] * ndim))

    if flatten:
        # Reshape to (N, 3) where N = H*W*D
        coords = coords.reshape(-1, 3)

    # Create offsets
    offsets = torch.zeros(batch_size + 1, dtype=torch.long, device="cpu")
    offsets[1:] = torch.arange(1, batch_size + 1, device="cpu") * (H * W * D)

    return coords, offsets


def strided_grid_coords(
    vertices: Tensor,
    curr_grid_shape: Tuple[int, int, int],
    grid_shape: Tuple[int, int, int],
    batch_size: int,
) -> Tensor:
    """Get vertices at a different resolution through striding or creating new coordinates.

    Args:
        vertices: Input coordinates tensor (B*H*W*D, 3)
        curr_grid_shape: Current grid shape (H, W, D)
        grid_shape: Target resolution (H, W, D)
        batch_size: Batch size

    Returns:
        Tensor: Grid vertices at the requested resolution
    """
    if curr_grid_shape == grid_shape:
        return vertices

    # Get device from input vertices
    device = vertices.device

    # Compute the stride
    if (
        curr_grid_shape[0] % grid_shape[0] == 0
        and curr_grid_shape[1] % grid_shape[1] == 0
        and curr_grid_shape[2] % grid_shape[2] == 0
    ):
        # If the current resolution is divisible by the target resolution,
        # we can use striding to get the vertices
        stride = (
            curr_grid_shape[0] // grid_shape[0],
            curr_grid_shape[1] // grid_shape[1],
            curr_grid_shape[2] // grid_shape[2],
        )

        # Reshape to 5D tensor (B, H, W, D, 3) for striding
        B = batch_size
        H, W, D = curr_grid_shape
        vertices_5d = vertices.reshape(B, H, W, D, 3)

        # Apply stride
        strided_vertices = vertices_5d[:, :: stride[0], :: stride[1], :: stride[2], :]

        # Reshape back to (B*H_out*W_out*D_out, 3)
        H_out, W_out, D_out = grid_shape
        strided_vertices = strided_vertices.reshape(B * H_out * W_out * D_out, 3)

        return strided_vertices
    else:
        # For non-divisible resolutions, create new coordinates from scratch
        # Get the bounds from original vertices for consistency
        B = batch_size
        H, W, D = curr_grid_shape
        vertices_5d = vertices.reshape(B, H, W, D, 3)

        # Get min and max bounds from the original vertices
        min_bound = torch.min(vertices_5d.view(-1, 3), dim=0)[0]
        max_bound = torch.max(vertices_5d.view(-1, 3), dim=0)[0]

        # Create new grid with the target resolution
        new_coords, _ = create_grid_coordinates(
            grid_shape=grid_shape,
            bounds=(min_bound, max_bound),
            batch_size=batch_size,
            device=device,
        )

        return new_coords
