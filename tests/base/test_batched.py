# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from warpconvnet.geometry.base.batched import BatchedTensor


@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing."""
    return [
        torch.randn(3, 2),  # 3x2 tensor
        torch.randn(4, 2),  # 4x2 tensor
        torch.randn(2, 2),  # 2x2 tensor
    ]


def test_init_from_list(sample_tensors):
    """Test initialization from a list of tensors."""
    batched = BatchedTensor(sample_tensors)
    assert batched.batch_size == 3
    assert batched.batched_tensor.shape == (9, 2)  # 3+4+2 = 9 rows
    assert len(batched.offsets) == 4  # n_tensors + 1
    assert torch.equal(batched.offsets, torch.tensor([0, 3, 7, 9]))


def test_init_from_tensor():
    """Test initialization from pre-concatenated tensor and offsets."""
    tensor = torch.randn(5, 2)
    offsets = [0, 2, 5]
    batched = BatchedTensor(tensor, offsets)
    assert batched.batch_size == 2
    assert torch.equal(batched.batched_tensor, tensor)
    assert torch.equal(batched.offsets, torch.tensor(offsets))


def test_getitem(sample_tensors):
    """Test tensor retrieval using indexing."""
    batched = BatchedTensor(sample_tensors)

    # Check each tensor in the batch
    for i in range(len(sample_tensors)):
        retrieved = batched[i]
        assert torch.equal(retrieved, sample_tensors[i])


def test_device_movement():
    """Test moving tensors between devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tensors = [torch.randn(3, 2), torch.randn(2, 2)]
    batched = BatchedTensor(tensors)

    # Move to GPU
    gpu_batched = batched.to("cuda")
    assert gpu_batched.batched_tensor.device.type == "cuda"

    # Move back to CPU
    cpu_batched = gpu_batched.to("cpu")
    assert cpu_batched.batched_tensor.device.type == "cpu"


def test_binary_operations(sample_tensors):
    """Test binary operations with scalars and other BatchedTensors."""
    batched = BatchedTensor(sample_tensors)

    # Test scalar multiplication
    scaled = batched.binary_op(2.0, "__mul__")
    assert torch.allclose(scaled[0], sample_tensors[0] * 2.0)

    # Test addition with another BatchedTensor
    batched2 = BatchedTensor(sample_tensors)
    added = batched.binary_op(batched2, "__add__")
    assert torch.allclose(added[0], sample_tensors[0] + sample_tensors[0])


def test_nested_tensor_conversion(sample_tensors):
    """Test conversion to and from nested tensors."""
    batched = BatchedTensor(sample_tensors)

    # Convert to nested tensor
    nested = batched.to_nested()
    assert isinstance(nested, torch.Tensor)
    assert nested.is_nested

    # Convert back to BatchedTensor
    converted = BatchedTensor.from_nested(nested)
    assert isinstance(converted, BatchedTensor)

    # Verify contents are preserved
    for i in range(len(sample_tensors)):
        assert torch.allclose(converted[i], sample_tensors[i])


def test_invalid_inputs():
    """Test error handling for invalid inputs."""
    # Test invalid tensor list
    with pytest.raises(RuntimeError):
        BatchedTensor([torch.randn(3, 2), torch.randn(3, 3)])  # Mismatched dimensions

    # Test invalid index
    batched = BatchedTensor([torch.randn(3, 2)])
    with pytest.raises(AssertionError):
        _ = batched[1.5]  # Non-integer index

    # Test invalid binary operation
    with pytest.raises(AssertionError):
        batched.binary_op(BatchedTensor([torch.randn(2, 2)]), "__add__")  # Mismatched sizes
