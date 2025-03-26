# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp

from warpconvnet.geometry.features.grid import GridFeatures, GridMemoryFormat
from warpconvnet.geometry.coords.grid import GridCoords
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.grid import Grid, points_to_grid
from warpconvnet.geometry.types.factorized_grid import FactorizedGrid

NUM_CHANNELS = 7


@pytest.fixture
def setup_device():
    """Setup test device."""
    wp.init()
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def setup_point_geometry(setup_device):
    """Setup test point geometry."""
    device = setup_device
    B, min_N, max_N, C = 3, 1000, 10000, NUM_CHANNELS
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Points(coords, features, device=device), Ns


def test_grid_init(setup_device):
    """Test Grid initialization."""
    device = setup_device
    grid_shape = (4, 6, 8)
    batch_size = 2

    # Create from grid shape
    geometry = Grid.create_from_grid_shape(
        grid_shape,
        NUM_CHANNELS,
        memory_format=GridMemoryFormat.b_x_y_z_c,
        batch_size=batch_size,
        device=device,
    )

    # Check properties
    assert geometry.grid_shape == grid_shape
    assert geometry.num_channels == NUM_CHANNELS
    assert geometry.memory_format == GridMemoryFormat.b_x_y_z_c
    assert geometry.batch_size == batch_size

    # Check device moving
    if torch.cuda.is_available():
        cpu_geometry = geometry.to(torch.device("cpu"))
        assert cpu_geometry.device == torch.device("cpu")
        assert geometry.device == device  # Original unchanged

    # Test memory format conversion
    for fmt in GridMemoryFormat:
        converted = geometry.to_memory_format(fmt)
        assert converted.memory_format == fmt
        assert converted.grid_shape == grid_shape
        assert converted.num_channels == NUM_CHANNELS


def test_factorized_grid_init(setup_device):
    """Test FactorizedGrid initialization."""
    device = setup_device
    # Create a list of grid shapes
    grid_shapes = [(2, 32, 64), (16, 2, 64), (16, 32, 2)]
    batch_size = 2

    # Create from grid shape
    factorized = FactorizedGrid.create_from_grid_shape(
        grid_shapes,
        NUM_CHANNELS,
        memory_formats=[
            GridMemoryFormat.b_zc_x_y,
            GridMemoryFormat.b_xc_y_z,
            GridMemoryFormat.b_yc_x_z,
        ],
        batch_size=batch_size,
        device=device,
    )

    # Check properties
    assert factorized.num_channels == NUM_CHANNELS
    assert factorized.batch_size == batch_size
    assert len(factorized) == 3

    # Check each geometry
    for i, fmt in enumerate(
        [GridMemoryFormat.b_zc_x_y, GridMemoryFormat.b_xc_y_z, GridMemoryFormat.b_yc_x_z]
    ):
        assert factorized[i].memory_format == fmt
        assert factorized[i].grid_shape == grid_shapes[i]
        assert factorized[i].num_channels == NUM_CHANNELS

    # Test get_by_format
    z_geometry = factorized.get_by_format(GridMemoryFormat.b_zc_x_y)
    assert z_geometry is not None
    assert z_geometry.memory_format == GridMemoryFormat.b_zc_x_y

    # Test with non-existent format
    standard_geometry = factorized.get_by_format(GridMemoryFormat.b_x_y_z_c)
    assert standard_geometry is None


def test_grid_creation_with_tensor(setup_device):
    """Test creating Grid from raw tensors."""
    device = setup_device
    grid_shape = (4, 6, 8)
    num_channels = NUM_CHANNELS
    batch_size = 2
    H, W, D = grid_shape

    # Create offsets
    elements_per_batch = H * W * D
    offsets = torch.tensor([0, elements_per_batch, 2 * elements_per_batch], device=device)

    # Create grid coordinates
    coords = GridCoords.create_regular_grid(grid_shape, batch_size=batch_size, device=device)

    # Test with standard format tensor
    tensor = torch.rand(batch_size, H, W, D, num_channels, device=device)
    geometry = Grid(coords, tensor, memory_format=GridMemoryFormat.b_x_y_z_c)

    assert geometry.grid_shape == grid_shape
    assert geometry.num_channels == num_channels
    assert geometry.memory_format == GridMemoryFormat.b_x_y_z_c
    assert geometry.batch_size == batch_size

    # Test with factorized format tensor
    tensor = torch.rand(batch_size, D * num_channels, H, W, device=device)
    geometry = Grid(
        coords,
        tensor,
        memory_format=GridMemoryFormat.b_zc_x_y,
        grid_shape=grid_shape,
        num_channels=num_channels,
    )

    assert geometry.grid_shape == grid_shape
    assert geometry.num_channels == num_channels
    assert geometry.memory_format == GridMemoryFormat.b_zc_x_y
    assert geometry.batch_size == batch_size


def test_factorized_grid_validation(setup_device):
    """Test FactorizedGrid validation."""
    device = setup_device
    grid_shape = (4, 6, 8)
    num_channels = 16
    batch_size = 2

    # Create standard geometry
    standard_geo = Grid.create_from_grid_shape(
        grid_shape,
        num_channels,
        memory_format=GridMemoryFormat.b_x_y_z_c,
        batch_size=batch_size,
        device=device,
    )

    # Create geometries with different formats
    geometries = [
        standard_geo.to_memory_format(GridMemoryFormat.b_zc_x_y),
        standard_geo.to_memory_format(GridMemoryFormat.b_xc_y_z),
        standard_geo.to_memory_format(GridMemoryFormat.b_yc_x_z),
    ]

    # Test valid creation
    factorized = FactorizedGrid(geometries)
    assert len(factorized) == 3

    # Test creating with duplicate formats - should raise an error
    geometries_with_duplicate = [
        standard_geo.to_memory_format(GridMemoryFormat.b_zc_x_y),
        standard_geo.to_memory_format(GridMemoryFormat.b_zc_x_y),  # Duplicate format
        standard_geo.to_memory_format(GridMemoryFormat.b_yc_x_z),
    ]

    with pytest.raises(AssertionError):
        FactorizedGrid(geometries_with_duplicate)

    # Test creating with non-factorized format - should raise an error
    geometries_with_standard = [
        standard_geo,  # Standard format (not factorized)
        standard_geo.to_memory_format(GridMemoryFormat.b_xc_y_z),
        standard_geo.to_memory_format(GridMemoryFormat.b_yc_x_z),
    ]

    with pytest.raises(AssertionError):
        FactorizedGrid(geometries_with_standard)


def test_factorized_grid_shapes(setup_device):
    """Test the shapes property of FactorizedGrid."""
    device = setup_device
    grid_shapes = [(2, 32, 64), (16, 2, 64), (16, 32, 2)]
    num_channels = NUM_CHANNELS
    batch_size = 2

    # Create factorized grid
    factorized = FactorizedGrid.create_from_grid_shape(
        grid_shapes,
        num_channels,
        memory_formats=[
            GridMemoryFormat.b_zc_x_y,
            GridMemoryFormat.b_xc_y_z,
            GridMemoryFormat.b_yc_x_z,
        ],
        batch_size=batch_size,
        device=device,
    )

    # Check shapes property
    shapes = factorized.shapes
    assert len(shapes) == 3

    for shape, grid_shape in zip(shapes, grid_shapes):
        assert shape["grid_shape"] == grid_shape
        assert shape["batch_size"] == batch_size
        assert shape["num_channels"] == num_channels
        assert (
            shape["total_elements"] == grid_shape[0] * grid_shape[1] * grid_shape[2] * batch_size
        )


def test_points_to_grid(setup_point_geometry):
    """Test points_to_grid function."""
    points, Ns = setup_point_geometry
    grid_shape = (4, 6, 8)

    # Test with different memory formats
    for memory_format in [GridMemoryFormat.b_x_y_z_c, GridMemoryFormat.b_c_x_y_z]:
        grid = points_to_grid(
            points,
            grid_shape,
            memory_format=memory_format,
            search_radius=0.1,
            search_type="radius",
            reduction="mean",
        )

        assert isinstance(grid, Grid)
        assert grid.memory_format == memory_format
        assert grid.grid_shape == grid_shape
        assert grid.num_channels == NUM_CHANNELS
        assert grid.batch_size == points.batch_size
