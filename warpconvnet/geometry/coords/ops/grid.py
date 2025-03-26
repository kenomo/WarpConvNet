# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from jaxtyping import Float, Int

import torch
from torch import Tensor


def create_grid_coordinates(
    grid_shape: Tuple[int, int, int],
    bounds: Optional[Tuple[Float[Tensor, "3"], Float[Tensor, "3"]]] = None,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> Tuple[Float[Tensor, "N 3"], Int[Tensor, "B+1"]]:  # noqa: F821
    """Create coordinate tensor for a regular grid.

    Args:
        grid_shape: Grid resolution (H, W, D)
        bounds: Min (xyz) and max (xyz) bounds (default: unit cube)
        batch_size: Number of batches
        device: Device to create tensors on

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

    # Reshape to (N, 3) where N = H*W*D
    coords = coords.reshape(-1, 3)

    # Create batched coords
    if batch_size > 1:
        coords = coords.repeat(batch_size, 1)

    # Create offsets
    offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.arange(1, batch_size + 1, device=device) * (H * W * D)

    return coords, offsets
