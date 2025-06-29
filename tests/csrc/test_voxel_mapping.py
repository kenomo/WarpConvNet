# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import pytest
import numpy as np
import torch
import time

import warpconvnet._C as _C


def points_to_closest_voxel_python(points, offsets, grid_shape, bounds_min, bounds_max):
    """Python reference implementation for points_to_closest_voxel_mapping."""
    num_points = points.shape[0]
    batch_size = offsets.shape[0] - 1
    voxel_indices = torch.zeros(num_points, dtype=torch.int32, device=points.device)

    # Compute grid sizes
    grid_size = (bounds_max - bounds_min) / grid_shape

    for idx in range(num_points):
        # Find batch
        batch_idx = 0
        for b in range(batch_size):
            if idx >= offsets[b] and idx < offsets[b + 1]:
                batch_idx = b
                break

        # Get point coordinates
        p = points[idx]

        # Compute voxel indices
        voxel_idx = ((p - bounds_min) / grid_size).floor().long()

        # Clamp to grid bounds
        voxel_idx[0] = torch.clamp(voxel_idx[0], min=0, max=grid_shape[0] - 1)
        voxel_idx[1] = torch.clamp(voxel_idx[1], min=0, max=grid_shape[1] - 1)
        voxel_idx[2] = torch.clamp(voxel_idx[2], min=0, max=grid_shape[2] - 1)

        # Compute flattened index
        flat_idx = (
            voxel_idx[0] * grid_shape[1] * grid_shape[2]
            + voxel_idx[1] * grid_shape[2]
            + voxel_idx[2]
        )

        # Add batch offset
        batch_voxel_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
        voxel_indices[idx] = batch_idx * batch_voxel_size + flat_idx

    return voxel_indices


def create_test_points(batch_sizes, bounds_min, bounds_max, device="cuda", seed=42):
    """Create test point cloud data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Ensure bounds are on the correct device
    bounds_min = bounds_min.to(device)
    bounds_max = bounds_max.to(device)

    # Create points for each batch
    all_points = []
    offsets = [0]

    for batch_size in batch_sizes:
        # Generate random points within bounds
        points = torch.rand(batch_size, 3, device=device)
        points = points * (bounds_max - bounds_min) + bounds_min
        all_points.append(points)
        offsets.append(offsets[-1] + batch_size)

    # Concatenate all points
    points = torch.cat(all_points, dim=0)
    offsets = torch.tensor(offsets, dtype=torch.int32, device=device)

    return points, offsets


def test_basic_functionality():
    """Test basic functionality of points_to_closest_voxel_mapping."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test parameters
    batch_sizes = [100, 150, 200]
    grid_shape = torch.tensor([8, 8, 8], dtype=torch.int32)
    bounds_min = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    bounds_max = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)

    # Create test data
    points, offsets = create_test_points(batch_sizes, bounds_min, bounds_max, device)

    # Run CUDA implementation
    cuda_result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape, bounds_min, bounds_max
    )

    # Run Python reference implementation
    python_result = points_to_closest_voxel_python(
        points, offsets, grid_shape.to(device), bounds_min.to(device), bounds_max.to(device)
    )

    # Compare results
    assert torch.equal(cuda_result, python_result), "CUDA and Python results don't match"
    print("✓ Basic functionality test passed")


@pytest.mark.parametrize("grid_shape", [(4, 4, 4), (16, 16, 16), (32, 32, 32), (128, 64, 32)])
@pytest.mark.parametrize("num_batches", [1, 4, 16])
def test_different_grid_shapes(grid_shape, num_batches):
    """Test with different grid shapes and batch sizes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test data
    batch_sizes = [100] * num_batches
    grid_shape_tensor = torch.tensor(grid_shape, dtype=torch.int32)
    bounds_min = torch.tensor([-5.0, -5.0, -5.0], dtype=torch.float32)
    bounds_max = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32)

    points, offsets = create_test_points(batch_sizes, bounds_min, bounds_max, device)

    # Run CUDA implementation
    cuda_result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape_tensor, bounds_min, bounds_max
    )

    # Run Python reference
    python_result = points_to_closest_voxel_python(
        points, offsets, grid_shape_tensor.to(device), bounds_min.to(device), bounds_max.to(device)
    )

    # Compare results
    assert torch.equal(
        cuda_result, python_result
    ), f"Results don't match for grid_shape={grid_shape}"
    print(f"✓ Grid shape {grid_shape} with {num_batches} batches test passed")


def test_edge_cases():
    """Test edge cases like points on boundaries, single point, etc."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    grid_shape = torch.tensor([4, 4, 4], dtype=torch.int32)
    bounds_min = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    bounds_max = torch.tensor([4.0, 4.0, 4.0], dtype=torch.float32)

    # Test 1: Single point at origin
    points = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    offsets = torch.tensor([0, 1], device=device, dtype=torch.int32)

    result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape, bounds_min, bounds_max
    )
    assert result[0] == 0, "Point at origin should map to voxel 0"

    # Test 2: Point at max bounds
    points = torch.tensor([[4.0, 4.0, 4.0]], device=device, dtype=torch.float32)
    result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape, bounds_min, bounds_max
    )
    expected_idx = 3 * 4 * 4 + 3 * 4 + 3  # Last voxel
    assert result[0] == expected_idx, f"Point at max bounds should map to voxel {expected_idx}"

    # Test 3: Points outside bounds (should be clamped)
    points = torch.tensor(
        [
            [-1.0, -1.0, -1.0],  # Below min
            [5.0, 5.0, 5.0],  # Above max
        ],
        device=device,
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 2], device=device, dtype=torch.int32)

    result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape, bounds_min, bounds_max
    )
    assert result[0] == 0, "Point below min bounds should clamp to voxel 0"
    assert (
        result[1] == expected_idx
    ), f"Point above max bounds should clamp to voxel {expected_idx}"

    print("✓ Edge cases test passed")


def test_large_scale():
    """Test with large point clouds to check performance and correctness."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for large scale test")

    device = "cuda"

    # Large scale parameters
    num_points_per_batch = 100000
    num_batches = 4
    batch_sizes = [num_points_per_batch] * num_batches

    grid_shape = torch.tensor([64, 64, 64], dtype=torch.int32)
    bounds_min = torch.tensor([-10.0, -10.0, -10.0], dtype=torch.float32)
    bounds_max = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)

    # Create test data
    points, offsets = create_test_points(batch_sizes, bounds_min, bounds_max, device)

    # Warm up
    for _ in range(3):
        _ = _C.utils.points_to_closest_voxel_mapping(
            points, offsets, grid_shape, bounds_min, bounds_max
        )

    # Time CUDA implementation
    torch.cuda.synchronize()
    start_time = time.time()

    cuda_result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape, bounds_min, bounds_max
    )

    torch.cuda.synchronize()
    cuda_time = time.time() - start_time

    total_points = num_points_per_batch * num_batches
    print(f"✓ Large scale test passed: {total_points:,} points in {cuda_time:.3f}s")
    print(f"  Throughput: {total_points/cuda_time/1e6:.2f} million points/sec")

    # Verify some basic properties
    assert cuda_result.shape[0] == total_points
    assert cuda_result.min() >= 0
    assert cuda_result.max() < num_batches * grid_shape.prod()


def test_float64_support():
    """Test that float64 (double) is supported."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"

    # Create float64 test data
    points = torch.rand(100, 3, device=device, dtype=torch.float64)
    offsets = torch.tensor([0, 50, 100], device=device, dtype=torch.int32)
    grid_shape = torch.tensor([8, 8, 8], dtype=torch.int32)
    bounds_min = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    bounds_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    # Should run without errors
    result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape, bounds_min, bounds_max
    )

    assert result.shape[0] == 100
    print("✓ Float64 support test passed")


if __name__ == "__main__":
    # Run basic tests if executed directly
    if torch.cuda.is_available():
        print("Running voxel mapping tests...")
        test_basic_functionality()
        test_edge_cases()
        test_large_scale()
        test_float64_support()
        print("\nAll tests passed! ✓")
    else:
        print("CUDA not available, skipping tests")
