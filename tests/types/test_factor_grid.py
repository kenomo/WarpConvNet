# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp

from warpconvnet.geometry.features.grid import GridMemoryFormat
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.grid import Grid
from warpconvnet.geometry.types.factor_grid import FactorGrid, points_to_factor_grid

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


def test_factorized_grid_init(setup_device):
    """Test FactorizedGrid initialization."""
    device = setup_device
    # Create a list of grid shapes
    grid_shapes = [(2, 32, 64), (16, 2, 64), (16, 32, 2)]
    batch_size = 2

    # Create from grid shape
    factorized = FactorGrid.create_from_grid_shape(
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
    factorized = FactorGrid(geometries)
    assert len(factorized) == 3

    # Test creating with duplicate formats - should raise an error
    geometries_with_duplicate = [
        standard_geo.to_memory_format(GridMemoryFormat.b_zc_x_y),
        standard_geo.to_memory_format(GridMemoryFormat.b_zc_x_y),  # Duplicate format
        standard_geo.to_memory_format(GridMemoryFormat.b_yc_x_z),
    ]

    with pytest.raises(AssertionError):
        FactorGrid(geometries_with_duplicate)

    # Test creating with non-factorized format - should raise an error
    geometries_with_standard = [
        standard_geo,  # Standard format (not factorized)
        standard_geo.to_memory_format(GridMemoryFormat.b_xc_y_z),
        standard_geo.to_memory_format(GridMemoryFormat.b_yc_x_z),
    ]

    with pytest.raises(AssertionError):
        FactorGrid(geometries_with_standard)


def test_factorized_grid_shapes(setup_device):
    """Test the shapes property of FactorizedGrid."""
    device = setup_device
    grid_shapes = [(2, 32, 64), (16, 2, 64), (16, 32, 2)]
    num_channels = NUM_CHANNELS
    batch_size = 2

    # Create factorized grid
    factorized = FactorGrid.create_from_grid_shape(
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


def test_points_to_factor_grid(setup_point_geometry):
    """Test points_to_factor_grid."""
    points, Ns = setup_point_geometry
    grid_shapes = [(2, 32, 64), (16, 2, 64), (16, 32, 2)]
    memory_formats = [
        GridMemoryFormat.b_zc_x_y,
        GridMemoryFormat.b_xc_y_z,
        GridMemoryFormat.b_yc_x_z,
    ]

    factorized = points_to_factor_grid(
        points,
        grid_shapes,
        memory_formats,
        search_radius=0.1,
        search_type="radius",
        reduction="mean",
    )
    assert isinstance(factorized, FactorGrid)
    assert len(factorized) == 3
    for i, fmt in enumerate(memory_formats):
        assert factorized[i].memory_format == fmt
        assert factorized[i].grid_shape == grid_shapes[i]
