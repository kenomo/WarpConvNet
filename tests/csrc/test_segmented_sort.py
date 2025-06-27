# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import pytest
import numpy as np
import torch

import warpconvnet._C as _C


def create_test_data(device="cuda", dtype=torch.float32, seed=42):
    """Create test data for segmented sort testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create segmented data: 3 segments with different sizes
    segment_sizes = [5, 3, 4]
    total_size = sum(segment_sizes)

    # Create random keys to sort
    keys = torch.randperm(total_size, device=device, dtype=torch.int64)

    # Create random values associated with keys
    if dtype == torch.int64:
        values = torch.randint(0, 100, (total_size,), device=device, dtype=dtype)
    else:
        values = torch.randn(total_size, device=device, dtype=dtype)

    # Create segment offsets [0, 5, 8, 12]
    segment_offsets = torch.tensor(
        [0] + [sum(segment_sizes[: i + 1]) for i in range(len(segment_sizes))],
        device=device,
        dtype=torch.int64,
    )

    return keys, values, segment_offsets, segment_sizes


def verify_segmented_sort(keys, sorted_keys, segment_offsets, descending=False):
    """Verify that segmented sort results are correct."""
    num_segments = len(segment_offsets) - 1

    for i in range(num_segments):
        start = segment_offsets[i]
        end = segment_offsets[i + 1]

        # Get original and sorted segments
        original_segment = keys[start:end]
        sorted_segment = sorted_keys[start:end]

        # Sort original segment for comparison
        expected_sorted = torch.sort(original_segment, descending=descending)[0]

        assert torch.equal(
            sorted_segment, expected_sorted
        ), f"Segment {i} not sorted correctly. Got: {sorted_segment}, Expected: {expected_sorted}"


def verify_sort_by_key(
    keys, values, sorted_values, sorted_keys, segment_offsets, descending=False
):
    """Verify that sort-by-key results are correct."""
    num_segments = len(segment_offsets) - 1

    for i in range(num_segments):
        start = segment_offsets[i]
        end = segment_offsets[i + 1]

        # Get segments
        key_segment = keys[start:end]
        value_segment = values[start:end]
        sorted_key_segment = sorted_keys[start:end]
        sorted_value_segment = sorted_values[start:end]

        # Sort by keys for comparison
        sorted_indices = torch.argsort(key_segment, descending=descending)
        expected_sorted_keys = key_segment[sorted_indices]
        expected_sorted_values = value_segment[sorted_indices]

        assert torch.equal(
            sorted_key_segment, expected_sorted_keys
        ), f"Segment {i} keys not sorted correctly"
        assert torch.equal(
            sorted_value_segment, expected_sorted_values
        ), f"Segment {i} values not sorted correctly"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("descending", [False, True])
def test_segmented_sort_keys_only(dtype, descending):
    """Test segmented sorting of keys only."""
    keys, values, segment_offsets, _ = create_test_data(dtype=dtype)

    # Sort keys only
    sorted_keys = _C.utils.segmented_sort(keys, segment_offsets, descending=descending)

    # Verify results
    verify_segmented_sort(keys, sorted_keys, segment_offsets, descending=descending)


@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("descending", [False, True])
def test_segmented_sort_by_key(value_dtype, descending):
    """Test segmented sorting of values by keys."""
    keys, values, segment_offsets, _ = create_test_data(dtype=value_dtype)

    # Sort values by keys
    sorted_values, sorted_keys = _C.utils.segmented_sort(
        keys, segment_offsets, values=values, descending=descending
    )

    # Verify results
    verify_sort_by_key(
        keys, values, sorted_values, sorted_keys, segment_offsets, descending=descending
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("descending", [False, True])
def test_segmented_sort_with_indices(dtype, descending):
    """Test segmented sorting returning permutation indices."""
    keys, values, segment_offsets, _ = create_test_data(dtype=dtype)

    # Sort keys only with indices
    perm_indices, sorted_keys = _C.utils.segmented_sort(
        keys, segment_offsets, descending=descending, return_indices=True
    )

    # Verify keys are sorted correctly
    verify_segmented_sort(keys, sorted_keys, segment_offsets, descending=descending)

    # Verify permutation indices are correct
    reconstructed_keys = keys[perm_indices]
    assert torch.equal(
        reconstructed_keys, sorted_keys
    ), "Permutation indices don't correctly reconstruct sorted keys"


@pytest.mark.parametrize("value_dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("descending", [False, True])
def test_segmented_sort_by_key_with_indices(value_dtype, descending):
    """Test segmented sorting by key returning permutation indices."""
    keys, values, segment_offsets, _ = create_test_data(dtype=value_dtype)

    # Sort by key with indices
    sorted_values, sorted_keys, perm_indices = _C.utils.segmented_sort(
        keys, segment_offsets, values=values, descending=descending, return_indices=True
    )

    # Verify sort results
    verify_sort_by_key(
        keys, values, sorted_values, sorted_keys, segment_offsets, descending=descending
    )

    # Verify permutation indices
    reconstructed_keys = keys[perm_indices]
    reconstructed_values = values[perm_indices]
    assert torch.equal(
        reconstructed_keys, sorted_keys
    ), "Permutation indices don't correctly reconstruct sorted keys"
    assert torch.equal(
        reconstructed_values, sorted_values
    ), "Permutation indices don't correctly reconstruct sorted values"


def test_empty_segments():
    """Test handling of empty segments."""
    device = "cuda"

    # Create data with some empty segments: [3, 0, 2, 0, 1]
    keys = torch.tensor([3, 1, 2, 7, 5, 9], device=device, dtype=torch.int64)
    segment_offsets = torch.tensor([0, 3, 3, 5, 5, 6], device=device, dtype=torch.int64)

    # Sort keys only
    sorted_keys = _C.utils.segmented_sort(keys, segment_offsets)

    expected = torch.tensor([1, 2, 3, 5, 7, 9], device=device, dtype=torch.int64)
    assert torch.equal(sorted_keys, expected), f"Got: {sorted_keys}, Expected: {expected}"


def test_single_element_segments():
    """Test segments with single elements."""
    device = "cuda"

    # Each segment has exactly one element
    keys = torch.tensor([5, 2, 8, 1], device=device, dtype=torch.int64)
    segment_offsets = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.int64)

    # Sort keys only - should remain unchanged
    sorted_keys = _C.utils.segmented_sort(keys, segment_offsets)

    assert torch.equal(sorted_keys, keys), "Single element segments should remain unchanged"


def test_large_data():
    """Stress test with larger data."""
    device = "cuda"
    torch.manual_seed(123)

    # Create larger segments
    segment_sizes = [1000, 500, 2000, 100]
    total_size = sum(segment_sizes)

    keys = torch.randperm(total_size, device=device, dtype=torch.int64)
    values = torch.randn(total_size, device=device, dtype=torch.float32)
    segment_offsets = torch.tensor(
        [0] + [sum(segment_sizes[: i + 1]) for i in range(len(segment_sizes))],
        device=device,
        dtype=torch.int64,
    )

    # Test all modes
    sorted_keys = _C.utils.segmented_sort(keys, segment_offsets)
    verify_segmented_sort(keys, sorted_keys, segment_offsets)

    sorted_values, sorted_keys = _C.utils.segmented_sort(keys, segment_offsets, values=values)
    verify_sort_by_key(keys, values, sorted_values, sorted_keys, segment_offsets)

    perm_indices, sorted_keys = _C.utils.segmented_sort(keys, segment_offsets, return_indices=True)
    verify_segmented_sort(keys, sorted_keys, segment_offsets)
    assert torch.equal(keys[perm_indices], sorted_keys)


def test_cpu_tensor_error():
    """Test that CPU tensors raise appropriate error."""
    keys = torch.tensor([3, 1, 2], dtype=torch.int64)  # CPU tensor
    segment_offsets = torch.tensor([0, 3], dtype=torch.int64)  # CPU tensor

    with pytest.raises(Exception):  # Should raise error for CPU tensors
        _C.utils.segmented_sort(keys, segment_offsets)


def test_mismatched_devices():
    """Test error handling for mismatched tensor devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    keys = torch.tensor([3, 1, 2], device="cuda", dtype=torch.int64)
    segment_offsets = torch.tensor([0, 3], dtype=torch.int64)  # CPU tensor

    with pytest.raises(Exception):  # Should raise error for mismatched devices
        _C.utils.segmented_sort(keys, segment_offsets)


@pytest.mark.parametrize("keys_dtype", [torch.int64])
def test_different_key_dtypes(keys_dtype):
    """Test with different key data types."""
    device = "cuda"
    torch.manual_seed(42)

    keys = torch.randperm(10, device=device, dtype=keys_dtype)
    segment_offsets = torch.tensor([0, 4, 7, 10], device=device, dtype=torch.int64)

    sorted_keys = _C.utils.segmented_sort(keys, segment_offsets)
    verify_segmented_sort(keys, sorted_keys, segment_offsets)


def test_int32_key_error():
    """Test that int32 keys raise appropriate error."""
    device = "cuda"
    keys = torch.randperm(10, device=device, dtype=torch.int32)
    segment_offsets = torch.tensor([0, 5, 10], device=device, dtype=torch.int64)

    with pytest.raises(Exception):  # Should raise error for int32 keys
        _C.utils.segmented_sort(keys, segment_offsets)


def test_identical_keys():
    """Test sorting segments with identical keys."""
    device = "cuda"

    # All keys are the same within each segment
    keys = torch.tensor([5, 5, 5, 2, 2, 8, 8, 8, 8], device=device, dtype=torch.int64)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device=device)
    segment_offsets = torch.tensor([0, 3, 5, 9], device=device, dtype=torch.int64)

    # Sort the data
    sorted_values, sorted_keys = _C.utils.segmented_sort(keys, segment_offsets, values=values)

    # Keys should be unchanged when all keys in each segment are identical
    assert torch.equal(sorted_keys, keys), "Identical keys should remain unchanged"

    # For identical keys, CUB uses stable sort, so values should preserve original order
    # within each segment
    assert torch.equal(
        sorted_values, values
    ), "Stable sort should preserve order for identical keys"


if __name__ == "__main__":
    # Run a quick test if executed directly
    if torch.cuda.is_available():
        print("Running basic segmented sort test...")
        test_segmented_sort_keys_only(torch.float32, False)
        test_segmented_sort_by_key(torch.float32, False)
        test_segmented_sort_with_indices(torch.float32, False)
        print("All basic tests passed!")
    else:
        print("CUDA not available, skipping tests")
