# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp
import torch

from warpconvnet.geometry.coords.search.warp_hashmap import (
    HashStruct,
    HashMethod as WarpHashMethod,
    WarpHashTable,
)


@pytest.fixture
def device():
    """Fixture for device configuration."""
    return "cuda:0"


@pytest.fixture
def sample_keys_warp(device):
    """Fixture providing sample vector keys as Warp arrays."""
    N = 48
    return wp.array(np.random.randint(0, 100, size=(N, 3), dtype=int), dtype=int, device=device)


def test_hash_struct_basic(device, sample_keys_warp):
    """Test basic HashStruct functionality."""
    capacity = 128

    # Initialize HashStruct
    hash_struct = HashStruct()
    hash_struct.capacity = capacity
    hash_struct.hash_method = WarpHashMethod.FNV1A.value

    # Test insertion
    hash_struct.insert(sample_keys_warp)

    # Test search
    search_results = hash_struct.search(sample_keys_warp)

    # Verify all keys are found
    assert (search_results.numpy() != -1).all(), "All keys should be found"


def test_hash_struct_serialization(device, sample_keys_warp):
    """Test HashStruct serialization and deserialization."""
    capacity = 128

    # Create and populate original struct
    original_struct = HashStruct()
    original_struct.capacity = capacity
    original_struct.hash_method = WarpHashMethod.FNV1A.value
    original_struct.insert(sample_keys_warp)

    # Save to dict
    data = original_struct.to_dict()

    # Load into new struct
    loaded_struct = HashStruct()
    loaded_struct.from_dict(data, device)

    # Verify search results match
    original_results = original_struct.search(sample_keys_warp)
    loaded_results = loaded_struct.search(sample_keys_warp)
    np.testing.assert_array_equal(
        original_results.numpy(),
        loaded_results.numpy(),
        "Search results should match after serialization",
    )


def test_vector_hash_table_creation(device, sample_keys_warp):
    """Test WarpVectorHashTable creation and basic operations."""
    # Test creation with capacity
    table = WarpHashTable(capacity=128, hash_method=WarpHashMethod.CITY)
    assert table._hash_struct is not None

    # Test creation from keys
    table_from_keys = WarpHashTable.from_keys(sample_keys_warp)
    assert table_from_keys._hash_struct is not None


def test_vector_hash_table_search(device, sample_keys_warp):
    """Test WarpVectorHashTable search functionality."""
    table = WarpHashTable.from_keys(sample_keys_warp)

    # Search for existing keys
    results = table.search(sample_keys_warp)
    assert (results.numpy() != -1).all(), "All existing keys should be found"

    # Search for non-existent keys
    invalid_keys = wp.array(
        np.random.randint(1000, 2000, size=(10, 3), dtype=int), dtype=int, device=device
    )
    invalid_results = table.search(invalid_keys)
    assert (invalid_results.numpy() == -1).all(), "Non-existent keys should not be found"


def test_hash_table_capacity_limits(device):
    """Test hash table behavior at capacity limits for WarpVectorHashTable."""
    small_capacity = 10
    table = WarpHashTable(capacity=small_capacity)

    # Try to insert more keys than capacity
    with pytest.raises(AssertionError):
        oversized_keys = wp.array(
            np.random.randint(0, 100, size=(small_capacity * 2, 3), dtype=int),
            dtype=int,
            device=device,
        )
        table._hash_struct.insert(oversized_keys)


def test_hash_methods_warp(device, sample_keys_warp):
    """Test different hash methods for WarpVectorHashTable."""
    for method in WarpHashMethod:
        table = WarpHashTable(capacity=128, hash_method=method)
        table._hash_struct.insert(sample_keys_warp)
        results = table._hash_struct.search(sample_keys_warp)
        assert (results.numpy() != -1).all(), f"Search failed for hash method {method}"


def test_torch_tensor_input_warp(device):
    """Test handling of torch tensor inputs for WarpVectorHashTable."""
    N = 48
    torch_keys = torch.randint(0, 100, (N, 3), device=device).int()

    # Test creation from torch tensor
    table = WarpHashTable.from_keys(torch_keys)

    # Test search with torch tensor
    results = table.search(torch_keys)
    assert (results.numpy() != -1).all(), "Search failed with torch tensor input"


if __name__ == "__main__":
    wp.init()
    pytest.main([__file__])
