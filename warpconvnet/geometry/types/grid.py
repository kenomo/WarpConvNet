# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Grid geometry implementation that combines grid coordinates and features.
"""

from typing import Dict, Literal, Optional, Tuple, Union

from jaxtyping import Float, Int
import torch
from torch import Tensor

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.coords.grid import GridCoords
from warpconvnet.geometry.coords.search.continuous import neighbor_search
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig, RealSearchMode
from warpconvnet.geometry.coords.search.search_results import RealSearchResult
from warpconvnet.geometry.features.grid import GridFeatures, GridMemoryFormat
from warpconvnet.geometry.types.points import Points


class Grid(Geometry):
    """Grid geometry representation that combines coordinates and features.

    This class provides a unified interface for grid-based geometries with any
    memory format, combining grid coordinates with grid features.

    Args:
        batched_coordinates (GridCoords): Coordinate system for the grid
        batched_features (Union[GridFeatures, Tensor]): Grid features
        memory_format (GridMemoryFormat): Memory format for the features
        grid_shape (Tuple[int, int, int], optional): Grid resolution (H, W, D)
        num_channels (int, optional): Number of feature channels
        **kwargs: Additional parameters
    """

    def __init__(
        self,
        batched_coordinates: GridCoords,
        batched_features: Union[GridFeatures, Tensor],
        memory_format: GridMemoryFormat = GridMemoryFormat.b_x_y_z_c,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        num_channels: Optional[int] = None,
        **kwargs,
    ):
        if isinstance(batched_features, Tensor):
            if grid_shape is None:
                grid_shape = batched_coordinates.grid_shape

            # If num_channels not provided, infer it from tensor shape and memory format
            if num_channels is None:
                if memory_format == GridMemoryFormat.b_x_y_z_c:
                    num_channels = batched_features.shape[-1]
                elif memory_format == GridMemoryFormat.b_c_x_y_z:
                    num_channels = batched_features.shape[1]
                else:
                    # For factorized formats, we need to infer from tensor shape and grid shape
                    if memory_format == GridMemoryFormat.b_zc_x_y:
                        zc = batched_features.shape[1]
                        num_channels = zc // grid_shape[2]
                    elif memory_format == GridMemoryFormat.b_xc_y_z:
                        xc = batched_features.shape[1]
                        num_channels = xc // grid_shape[0]
                    elif memory_format == GridMemoryFormat.b_yc_x_z:
                        yc = batched_features.shape[1]
                        num_channels = yc // grid_shape[1]
                    else:
                        raise ValueError(f"Unsupported memory format: {memory_format}")

            # Create GridFeatures with same offsets as coordinates
            batched_features = GridFeatures(
                batched_features,
                batched_coordinates.offsets.clone(),
                memory_format,
                grid_shape,
                num_channels,
            )

        # Ensure offsets match
        assert (
            batched_coordinates.offsets == batched_features.offsets
        ).all(), "Coordinate and feature offsets must match"

        # Initialize base class
        super().__init__(batched_coordinates, batched_features, **kwargs)
        self.memory_format = memory_format

    @classmethod
    def create_from_grid_shape(
        cls,
        grid_shape: Tuple[int, int, int],
        num_channels: int,
        memory_format: GridMemoryFormat = GridMemoryFormat.b_x_y_z_c,
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "Grid":
        """Create a new GridGeometry with initialized grid coordinates and features.

        Args:
            grid_shape: Grid resolution (H, W, D)
            num_channels: Number of feature channels
            memory_format: Memory format for features
            bounds: Min and max bounds for the grid
            batch_size: Number of batches
            device: Device to create tensors on
            dtype: Data type for feature tensors
            **kwargs: Additional parameters

        Returns:
            Initialized grid geometry
        """
        # Create coordinates
        coords = GridCoords.create_regular_grid(grid_shape, bounds, batch_size, device)

        # Create empty features with same offsets
        features = GridFeatures.create_empty(
            grid_shape, num_channels, batch_size, memory_format, device, dtype
        )

        # Make sure offsets match
        assert (
            coords.offsets == features.offsets
        ).all(), "Coordinate and feature offsets must match"

        return cls(coords, features, memory_format, **kwargs)

    @property
    def grid_features(self) -> GridFeatures:
        """Get the grid features."""
        return self.batched_features

    @property
    def grid_coords(self) -> GridCoords:
        """Get the grid coordinates."""
        return self.batched_coordinates

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Get the grid shape (H, W, D)."""
        return self.grid_coords.grid_shape

    @property
    def num_channels(self) -> int:
        """Get the number of feature channels."""
        return self.grid_features.num_channels

    def to_memory_format(self, memory_format: GridMemoryFormat) -> "Grid":
        """Convert to a different memory format."""
        if memory_format != self.memory_format:
            return self.replace(
                batched_features=self.grid_features.to_memory_format(memory_format),
                memory_format=memory_format,
            )
        return self

    @property
    def shape(self) -> Dict[str, Union[int, Tuple[int, ...]]]:
        """Get the shape information."""
        H, W, D = self.grid_shape
        return {
            "grid_shape": self.grid_shape,
            "batch_size": self.batch_size,
            "num_channels": self.num_channels,
            "total_elements": H * W * D * self.batch_size,
        }

    def to(self, device: torch.device) -> "Grid":
        """Move the geometry to the specified device."""
        return Grid(
            self.grid_coords.to(device),
            self.grid_features.to(device),
            self.memory_format,
        )


def _point_to_grid_mapping(
    point_coords: Float[Tensor, "N 3"],  # noqa: F821
    point_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    grid_coords: Float[Tensor, "M 3"],  # noqa: F821
    grid_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    search_radius: Optional[float] = None,
    k: Optional[int] = None,
    search_type: Literal["radius", "knn"] = "radius",
    search_grid_dim: Optional[int] = None,
) -> RealSearchResult:
    """Compute the mapping from points to grid voxels.

    Args:
        point_coords: Point coordinates (N, 3)
        point_offsets: Point offsets (B+1)
        grid_coords: Grid coordinates (M, 3)
        grid_offsets: Grid offsets (B+1)
        search_radius: Search radius for radius search
        k: Number of neighbors for kNN search
        search_type: Search type ('radius' or 'knn')
        search_grid_dim: Grid dimension for radius search

    Returns:
        RealSearchResult: Search results class
    """
    # Create search config
    if search_type == "radius":
        assert search_radius is not None, "Search radius must be provided for radius search"
        search_mode = RealSearchMode.RADIUS
        search_config = RealSearchConfig(
            mode=search_mode, radius=search_radius, grid_dim=search_grid_dim
        )
    elif search_type == "knn":
        search_mode = RealSearchMode.KNN
        search_config = RealSearchConfig(mode=search_mode, knn_k=k)
    else:
        raise ValueError(f"Unsupported search type: {search_type}")

    # Note: We're reversing the query and reference compared to the original function
    # as we want to find the points near each grid voxel
    results: RealSearchResult = neighbor_search(
        ref_positions=point_coords,
        ref_offsets=point_offsets,
        query_positions=grid_coords,
        query_offsets=grid_offsets,
        search_args=search_config,
    )

    return results


def points_to_grid_features(
    points: Points,
    grid_shape: Tuple[int, int, int],
    memory_format: GridMemoryFormat = GridMemoryFormat.b_x_y_z_c,
    bounds: Optional[Tuple[Tensor, Tensor]] = None,
    search_radius: Optional[float] = None,
    k: int = 8,
    search_type: str = "radius",
    reduction: str = "mean",
) -> "GridFeatures":
    """Convert point features to grid features.

    Args:
        points: Input point geometry
        grid_shape: Grid shape (H, W, D)
        memory_format: Memory format for grid features
        bounds: Min and max bounds for the grid
        search_radius: Search radius for radius search
        k: Number of neighbors for kNN search
        search_type: Search type ('radius' or 'knn')
        reduction: Reduction method ('mean', 'max', 'sum')

    Returns:
        GridFeatures: Grid features with the specified memory format
    """
    batch_size = points.batch_size
    device = points.device
    num_channels = points.num_channels

    # Create grid coordinates
    grid_coords = GridCoords.create_regular_grid(
        grid_shape,
        bounds=bounds,
        batch_size=batch_size,
        device=device,
    )

    # Map points to grid
    point_coords = points.batched_coordinates.batched_tensor
    point_offsets = points.batched_coordinates.offsets
    grid_coords_tensor = grid_coords.batched_tensor
    grid_offsets = grid_coords.offsets

    search_results = _point_to_grid_mapping(
        point_coords,
        point_offsets,
        grid_coords_tensor,
        grid_offsets,
        search_radius=search_radius,
        k=k,
        search_type=search_type,
        search_grid_dim=max(grid_shape),
    )

    # Get neighbor indices
    neighbor_indices = search_results.neighbor_indices

    # Gather point features
    point_features = points.batched_features.batched_tensor

    # Initialize grid features
    H, W, D = grid_shape
    total_grid_points = H * W * D

    if memory_format == GridMemoryFormat.b_x_y_z_c:
        grid_features = torch.zeros(
            (batch_size, H, W, D, num_channels),
            device=device,
            dtype=point_features.dtype,
        )
    else:
        # For other memory formats, we'll first create in standard format
        # and then convert
        grid_features = torch.zeros(
            (batch_size, H, W, D, num_channels),
            device=device,
            dtype=point_features.dtype,
        )

    # Process search results differently based on search type
    if search_type == "radius":
        # For radius search, we have a split array
        if hasattr(search_results, "neighbor_row_splits"):
            # Process each grid cell
            for grid_idx, start_idx in enumerate(search_results.neighbor_row_splits[:-1]):
                end_idx = search_results.neighbor_row_splits[grid_idx + 1]

                # Skip if no neighbors found
                if start_idx == end_idx:
                    continue

                # Get indices of neighboring points
                point_indices = neighbor_indices[start_idx:end_idx]

                # Get the corresponding point features
                neighbor_features = point_features[point_indices]

                # Calculate batch and cell indices
                batch_idx = torch.searchsorted(grid_coords.offsets, grid_idx, right=True) - 1
                cell_idx = grid_idx - grid_coords.offsets[batch_idx]

                # Compute grid indices
                h_idx = cell_idx // (W * D)
                w_idx = (cell_idx % (W * D)) // D
                d_idx = cell_idx % D

                # Apply reduction
                if reduction == "mean":
                    reduced_features = torch.mean(neighbor_features, dim=0)
                elif reduction == "max":
                    reduced_features = torch.max(neighbor_features, dim=0)[0]
                elif reduction == "sum":
                    reduced_features = torch.sum(neighbor_features, dim=0)
                elif reduction == "prod":
                    reduced_features = torch.prod(neighbor_features, dim=0)
                else:
                    raise ValueError(f"Unsupported reduction: {reduction}")

                # Set grid features
                grid_features[batch_idx, h_idx, w_idx, d_idx] = reduced_features
        else:
            # Fallback for older versions without row_splits
            raise NotImplementedError("Radius search without neighbor_row_splits is not supported")
    else:
        # For KNN search, we have a fixed number of neighbors per grid cell
        total_grid_cells = grid_coords.batched_tensor.shape[0]

        # Calculate batch and spatial indices for all grid cells
        # Make sure arange is on the same device as offsets
        batch_indices = (
            torch.searchsorted(
                grid_coords.offsets,
                torch.arange(total_grid_cells, device=grid_coords.offsets.device),
                right=True,
            )
            - 1
        )
        cell_indices = (
            torch.arange(total_grid_cells, device=grid_coords.offsets.device)
            - grid_coords.offsets[batch_indices]
        )

        # Compute 3D grid indices
        h_indices = cell_indices // (W * D)
        w_indices = (cell_indices % (W * D)) // D
        d_indices = cell_indices % D

        # Process each grid cell
        for i in range(total_grid_cells):
            # Get neighboring point indices
            point_indices = neighbor_indices[i]

            # Skip invalid indices
            valid_mask = point_indices >= 0
            if not torch.any(valid_mask):
                continue

            point_indices = point_indices[valid_mask]

            # Get point features
            neighbor_features = point_features[point_indices]

            # Apply reduction
            if reduction == "mean":
                reduced_features = torch.mean(neighbor_features, dim=0)
            elif reduction == "max":
                reduced_features = torch.max(neighbor_features, dim=0)[0]
            elif reduction == "sum":
                reduced_features = torch.sum(neighbor_features, dim=0)
            else:
                raise ValueError(f"Unsupported reduction: {reduction}")

            # Set grid features
            batch_idx = batch_indices[i]
            h_idx = h_indices[i]
            w_idx = w_indices[i]
            d_idx = d_indices[i]

            grid_features[batch_idx, h_idx, w_idx, d_idx] = reduced_features

    # Convert to target memory format if needed
    if memory_format != GridMemoryFormat.b_x_y_z_c:
        B = batch_size
        C = num_channels

        if memory_format == GridMemoryFormat.b_c_x_y_z:
            grid_features = grid_features.permute(0, 4, 1, 2, 3)
        elif memory_format == GridMemoryFormat.b_zc_x_y:
            grid_features = grid_features.permute(0, 3, 4, 1, 2).reshape(B, D * C, H, W)
        elif memory_format == GridMemoryFormat.b_xc_y_z:
            grid_features = grid_features.permute(0, 1, 4, 2, 3).reshape(B, H * C, W, D)
        elif memory_format == GridMemoryFormat.b_yc_x_z:
            grid_features = grid_features.permute(0, 2, 4, 1, 3).reshape(B, W * C, H, D)
        else:
            raise ValueError(f"Unsupported memory format: {memory_format}")

    # Create GridFeatures
    return GridFeatures(
        grid_features, grid_coords.offsets, memory_format, grid_shape, num_channels
    )


def points_to_grid(
    points: Points,
    grid_shape: Tuple[int, int, int],
    memory_format: GridMemoryFormat = GridMemoryFormat.b_x_y_z_c,
    bounds: Optional[Tuple[Tensor, Tensor]] = None,
    search_radius: Optional[float] = None,
    k: int = 8,
    search_type: str = "radius",
    reduction: str = "mean",
) -> "Grid":
    """Convert point features to a grid.

    Args:
        points: Input point geometry
        grid_shape: Grid shape (H, W, D)
        memory_format: Memory format for grid features
        bounds: Min and max bounds for the grid
        search_radius: Search radius for radius search
        k: Number of neighbors for kNN search
        search_type: Search type ('radius' or 'knn')
        reduction: Reduction method ('mean', 'max', 'sum')

    Returns:
        Grid: Grid with the specified memory format
    """
    # Create grid coordinates
    batch_size = points.batch_size
    device = points.device

    grid_coords = GridCoords.create_regular_grid(
        grid_shape,
        bounds=bounds,
        batch_size=batch_size,
        device=device,
    )

    # Convert point features to grid features
    grid_features = points_to_grid_features(
        points,
        grid_shape,
        memory_format=memory_format,
        bounds=bounds,
        search_radius=search_radius,
        k=k,
        search_type=search_type,
        reduction=reduction,
    )

    # Create and return grid geometry
    return Grid(grid_coords, grid_features, memory_format)
