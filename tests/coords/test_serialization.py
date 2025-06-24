# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.coords.ops.serialization import (
    encode,
    morton_code,
    POINT_ORDERING,
)


@pytest.fixture
def setup_coordinates():
    """Setup test coordinates for serialization testing."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create sample 3D coordinates
    batch_size = 3
    num_points = 100

    # Generate random coordinates in a reasonable range for Morton coding
    coords = torch.randint(0, 100, (num_points, 3), device=device, dtype=torch.int32)

    # Create batch indices
    batch_indices = torch.randint(0, batch_size, (num_points,), device=device)

    # Create offsets for batched processing
    offsets = torch.tensor([0, 30, 70, num_points], device=device)

    return coords, batch_indices, offsets, batch_size, device


@pytest.fixture
def setup_grid_coordinates():
    """Setup regular grid coordinates for deterministic testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a small regular grid
    x = torch.arange(4, device=device)
    y = torch.arange(4, device=device)
    z = torch.arange(4, device=device)

    # Create meshgrid and flatten
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

    return coords, device


def test_encode_with_different_orders(setup_grid_coordinates):
    """Test encoding with different coordinate orderings."""
    coords, device = setup_grid_coordinates

    # Test all supported orderings
    codes_by_order = {}
    for order in POINT_ORDERING:
        codes = encode(coords, order=order, return_perm=False)
        codes_by_order[order] = codes

        assert codes.shape == (coords.shape[0],), f"Codes shape mismatch for order {order}"
        assert codes.dtype == torch.int64, f"Codes dtype mismatch for order {order}"


def test_morton_code_single_batch(setup_coordinates):
    """Test morton_code function with single batch."""
    coords, _, _, _, device = setup_coordinates

    # Test without offsets (single batch)
    codes, ordering = morton_code(coords.float(), order=POINT_ORDERING.MORTON_XYZ)

    assert codes.shape == (coords.shape[0],), "Codes shape mismatch"
    assert ordering.shape == (coords.shape[0],), "Ordering shape mismatch"
    assert codes.dtype == torch.int64, "Codes should be int64"
    assert ordering.dtype == torch.int64, "Ordering should be int64"

    # Test that ordering actually sorts the codes
    sorted_codes = codes[ordering]
    assert torch.all(sorted_codes[1:] >= sorted_codes[:-1]), "Codes should be sorted by ordering"


def test_morton_code_multi_batch(setup_coordinates):
    """Test morton_code function with multiple batches."""
    coords, batch_indices, offsets, batch_size, device = setup_coordinates

    # Test with offsets (multi-batch)
    codes, ordering = morton_code(
        coords.float(), batch_offsets=offsets, order=POINT_ORDERING.MORTON_XYZ
    )

    assert codes.shape == (coords.shape[0],), "Codes shape mismatch for multi-batch"
    assert ordering.shape == (coords.shape[0],), "Ordering shape mismatch for multi-batch"

    # Test that ordering sorts within each batch
    for i in range(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        batch_codes = codes[start:end]
        batch_ordering = ordering[start:end] - start  # Adjust for batch offset

        sorted_batch_codes = batch_codes[batch_ordering]
        assert torch.all(
            sorted_batch_codes[1:] >= sorted_batch_codes[:-1]
        ), f"Batch {i} codes should be sorted"


def test_coordinate_permutation_effects(setup_grid_coordinates):
    """Test that coordinate permutations have the expected effects."""
    coords, device = setup_grid_coordinates

    # Test specific coordinate transformations
    # Original point: [1, 2, 3]
    test_coord = torch.tensor([[1, 2, 3]], device=device, dtype=torch.float32)

    # Test xyz vs yxz (swap x and y)
    code_xyz = encode(test_coord, order=POINT_ORDERING.MORTON_XYZ)

    # Manually swap coordinates and encode with xyz - should match yxz encoding
    swapped_coord = test_coord[:, [1, 0, 2]]  # [2, 1, 3]
    code_swapped = encode(swapped_coord, order=POINT_ORDERING.MORTON_XYZ)
    code_yxz = encode(test_coord, order=POINT_ORDERING.MORTON_YXZ)

    assert (
        code_swapped.item() == code_yxz.item()
    ), "yxz ordering should match manually swapped coordinates"


def test_ordering_deterministic():
    """Test that ordering is deterministic for the same input."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coords = torch.randint(0, 50, (20, 3), device=device, dtype=torch.float32)

    # Generate codes multiple times
    codes1 = encode(coords, order=POINT_ORDERING.MORTON_XYZ)
    codes2 = encode(coords, order=POINT_ORDERING.MORTON_XYZ)
    codes3 = encode(coords, order=POINT_ORDERING.MORTON_YXZ)
    codes4 = encode(coords, order=POINT_ORDERING.MORTON_YXZ)

    # Same order should give same results
    assert torch.equal(codes1, codes2), "xyz encoding should be deterministic"
    assert torch.equal(codes3, codes4), "yxz encoding should be deterministic"


def test_empty_coordinates():
    """Test handling of empty coordinate arrays."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    empty_coords = torch.empty((0, 3), device=device, dtype=torch.float32)

    # Test that empty coordinates don't crash
    codes = encode(empty_coords, order=POINT_ORDERING.MORTON_XYZ)
    assert codes.shape == (0,), "Empty coordinates should produce empty codes"

    # Test morton_code with empty coordinates
    codes, ordering = morton_code(empty_coords, order=POINT_ORDERING.MORTON_XYZ)
    assert codes.shape == (0,), "Empty coordinates should produce empty codes in morton_code"
    assert ordering.shape == (0,), "Empty coordinates should produce empty ordering"


def test_large_coordinates():
    """Test with coordinates at the edge of the supported range."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test coordinates near the maximum for 16-bit path
    large_coords = torch.tensor(
        [
            [0, 0, 0],
            [65535, 65535, 65535],  # Max for 16-bit
            [1000, 2000, 3000],
        ],
        device=device,
        dtype=torch.int32,
    )

    # Should not crash and should produce valid codes
    codes = encode(large_coords, order=POINT_ORDERING.MORTON_XYZ)
    assert codes.shape == (3,), "Large coordinates should be handled properly"

    # All codes should be different
    assert len(torch.unique(codes)) == 3, "Large coordinates should produce unique codes"
