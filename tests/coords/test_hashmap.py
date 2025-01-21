import numpy as np
import pytest
import warp as wp
import torch

from warpconvnet.geometry.coords.search.hashmap import (
    HashStruct,
    HashMethod,
    VectorHashTable,
)


@pytest.fixture
def device():
    """Fixture for device configuration."""
    return "cuda:0"


@pytest.fixture
def sample_keys(device):
    """Fixture providing sample vector keys."""
    N = 48
    return wp.array(np.random.randint(0, 100, size=(N, 3), dtype=int), dtype=int, device=device)


def test_hash_struct_basic(device, sample_keys):
    """Test basic HashStruct functionality."""
    capacity = 128

    # Initialize HashStruct
    hash_struct = HashStruct()
    hash_struct.capacity = capacity
    hash_struct.hash_method = HashMethod.FNV1A.value

    # Test insertion
    hash_struct.insert(sample_keys)

    # Test search
    search_results = hash_struct.search(sample_keys)

    # Verify all keys are found
    assert (search_results.numpy() != -1).all(), "All keys should be found"


def test_hash_struct_serialization(device, sample_keys):
    """Test HashStruct serialization and deserialization."""
    capacity = 128

    # Create and populate original struct
    original_struct = HashStruct()
    original_struct.capacity = capacity
    original_struct.hash_method = HashMethod.FNV1A.value
    original_struct.insert(sample_keys)

    # Save to dict
    data = original_struct.to_dict()

    # Load into new struct
    loaded_struct = HashStruct()
    loaded_struct.from_dict(data, device)

    # Verify search results match
    original_results = original_struct.search(sample_keys)
    loaded_results = loaded_struct.search(sample_keys)
    np.testing.assert_array_equal(
        original_results.numpy(),
        loaded_results.numpy(),
        "Search results should match after serialization",
    )


def test_vector_hash_table_creation(device, sample_keys):
    """Test VectorHashTable creation and basic operations."""
    # Test creation with capacity
    table = VectorHashTable(capacity=128, hash_method=HashMethod.CITY)
    assert table._hash_struct is not None

    # Test creation from keys
    table_from_keys = VectorHashTable.from_keys(sample_keys)
    assert table_from_keys._hash_struct is not None


def test_vector_hash_table_search(device, sample_keys):
    """Test VectorHashTable search functionality."""
    table = VectorHashTable.from_keys(sample_keys)

    # Search for existing keys
    results = table.search(sample_keys)
    assert (results.numpy() != -1).all(), "All existing keys should be found"

    # Search for non-existent keys
    invalid_keys = wp.array(
        np.random.randint(1000, 2000, size=(10, 3), dtype=int), dtype=int, device=device
    )
    invalid_results = table.search(invalid_keys)
    assert (invalid_results.numpy() == -1).all(), "Non-existent keys should not be found"


def test_hash_table_capacity_limits(device):
    """Test hash table behavior at capacity limits."""
    small_capacity = 10
    table = VectorHashTable(capacity=small_capacity)

    # Try to insert more keys than capacity
    with pytest.raises(AssertionError):
        oversized_keys = wp.array(
            np.random.randint(0, 100, size=(small_capacity * 2, 3), dtype=int),
            dtype=int,
            device=device,
        )
        table._hash_struct.insert(oversized_keys)


def test_hash_methods(device, sample_keys):
    """Test different hash methods."""
    for method in HashMethod:
        table = VectorHashTable(capacity=128, hash_method=method)
        table._hash_struct.insert(sample_keys)
        results = table._hash_struct.search(sample_keys)
        assert (results.numpy() != -1).all(), f"Search failed for hash method {method}"


def test_torch_tensor_input(device):
    """Test handling of torch tensor inputs."""
    N = 48
    torch_keys = torch.randint(0, 100, (N, 3), device=device).int()

    # Test creation from torch tensor
    table = VectorHashTable.from_keys(torch_keys)

    # Test search with torch tensor
    results = table.search(torch_keys)
    assert (results.numpy() != -1).all(), "Search failed with torch tensor input"


if __name__ == "__main__":
    wp.init()
    pytest.main([__file__])
