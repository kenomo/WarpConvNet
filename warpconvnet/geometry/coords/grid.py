# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from jaxtyping import Float, Int

import torch
from torch import Tensor

from warpconvnet.geometry.base.coords import Coords
from warpconvnet.geometry.coords.ops.grid import create_grid_coordinates


def grid_init(
    bb_max: Tuple[float, float, float],
    bb_min: Tuple[float, float, float],
    resolution: Tuple[int, int, int],
) -> Float[Tensor, "res[0] res[1] res[2] 3"]:  # noqa: F821
    """Initialize grid coordinates."""
    H, W, D = resolution
    x = torch.linspace(bb_min[0], bb_max[0], W)
    y = torch.linspace(bb_min[1], bb_max[1], H)
    z = torch.linspace(bb_min[2], bb_max[2], D)
    return torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)


class GridCoords(Coords):
    grid_shape: Tuple[int, int, int]
    min_bound: Float[Tensor, "3"]  # noqa: F821
    max_bound: Float[Tensor, "3"]  # noqa: F821

    def __init__(
        self,
        batched_tensor: Float[Tensor, "N 3"],  # noqa: F821
        offsets: Int[Tensor, "B+1"],  # noqa: F821
        grid_shape: Tuple[int, int, int],
        bounds: Optional[Tuple[Float[Tensor, "3"], Float[Tensor, "3"]]] = None,
    ):
        """
        Args:
            batched_tensor: provides the coordinates of the points
            offsets: provides the offsets for each batch
            grid_shape: provides the shape of the grid
            bounds: provides the min and max bounds of the grid
        """
        super().__init__(batched_tensor, offsets)
        self.grid_shape = grid_shape

        if bounds is None:
            # Default to unit cube
            self.min_bound = torch.zeros(3, device=batched_tensor.device)
            self.max_bound = torch.ones(3, device=batched_tensor.device)
        else:
            assert isinstance(bounds, tuple) and len(bounds) == 2
            self.min_bound, self.max_bound = bounds

    @classmethod
    def create_regular_grid(
        cls,
        grid_shape: Tuple[int, int, int],
        bounds: Optional[Tuple[Float[Tensor, "3"], Float[Tensor, "3"]]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> "GridCoords":
        """Create coordinates for a regular grid.

        Args:
            grid_shape (Tuple[int, int, int]): Grid resolution (H, W, D)
            bounds (Tuple[Tensor, Tensor]): Min and max bounds (default: unit cube)
            batch_size (int): Number of batches
            device (torch.device): Device to create tensors on

        Returns:
            GridCoords: Coordinates for a regular grid
        """
        assert (
            isinstance(grid_shape, tuple) and len(grid_shape) == 3
        ), f"grid_shape: {grid_shape} must be a tuple of 3 integers."
        coords, offsets = create_grid_coordinates(grid_shape, bounds, batch_size, device)

        return cls(coords, offsets, grid_shape, bounds)

    def resample(self, new_grid_shape: Tuple[int, int, int]) -> "GridCoords":
        """Resample grid coordinates to a new resolution.

        Args:
            new_grid_shape (Tuple[int, int, int]): New grid resolution

        Returns:
            GridCoords: Resampled coordinates
        """
        if new_grid_shape == self.grid_shape:
            return self

        # Create new grid coordinates
        coords, offsets = create_grid_coordinates(
            new_grid_shape, (self.min_bound, self.max_bound), len(self.offsets) - 1, self.device
        )

        return self.__class__(coords, offsets, new_grid_shape, (self.min_bound, self.max_bound))

    def get_spatial_indices(
        self, flat_indices: Int[Tensor, "M"]  # noqa: F821
    ) -> Tuple[Int[Tensor, "M"], Int[Tensor, "M"], Int[Tensor, "M"]]:  # noqa: F821
        """Convert flattened indices to 3D spatial indices.

        Args:
            flat_indices (Tensor): Flattened indices

        Returns:
            Tuple[Int[Tensor, "M"], Int[Tensor, "M"], Int[Tensor, "M"]]: H, W, D indices
        """
        H, W, D = self.grid_shape

        # Calculate indices for each dimension
        h_indices = flat_indices // (W * D)
        w_indices = (flat_indices % (W * D)) // D
        d_indices = flat_indices % D

        return h_indices, w_indices, d_indices

    def get_flattened_indices(
        self,
        h_indices: Int[Tensor, "M"],  # noqa: F821
        w_indices: Int[Tensor, "M"],  # noqa: F821
        d_indices: Int[Tensor, "M"],  # noqa: F821
    ) -> Int[Tensor, "M"]:  # noqa: F821
        """Convert 3D spatial indices to flattened indices.

        Args:
            h_indices (Int[Tensor, "M"]): H indices
            w_indices (Int[Tensor, "M"]): W indices
            d_indices (Int[Tensor, "M"]): D indices

        Returns:
            Int[Tensor, "M"]: Flattened indices
        """
        H, W, D = self.grid_shape

        return h_indices * (W * D) + w_indices * D + d_indices

    def to(self, device: torch.device) -> "GridCoords":
        """Move the coordinates to a specific device.

        Args:
            device (torch.device): Device to move the coordinates to

        Returns:
            GridCoords: Coordinates on the new device
        """
        return self.__class__(
            self.batched_tensor.to(device),
            self.offsets.to(device),
            self.grid_shape,
            (
                None
                if self.min_bound is None
                else (self.min_bound.to(device), self.max_bound.to(device))
            ),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(grid_shape={self.grid_shape}, min_bound={self.min_bound}, max_bound={self.max_bound}, batch_size={self.batch_size})"
