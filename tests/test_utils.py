# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp

from warpconvnet.utils.argsort import argsort
from warpconvnet.utils.unique import ToUnique, unique_torch


@pytest.fixture
def setup_data():
    """Setup test data."""
    wp.init()
    torch.manual_seed(0)
    return None


@pytest.mark.benchmark(group="argsort")
@pytest.mark.parametrize("backend", ["torch", "warp"])
@pytest.mark.parametrize("input_type", ["torch", "warp"])
def test_argsort(setup_data, benchmark, backend, input_type):
    """Benchmark argsort with different backends and input types."""
    device = "cuda:0"
    N = 1000000
    rand_perm = torch.randperm(N, device=device).int()

    if input_type == "warp":
        rand_perm_in = wp.from_torch(rand_perm)
    else:
        rand_perm_in = rand_perm

    result = benchmark.pedantic(
        lambda: argsort(rand_perm_in, backend=backend),
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )


def test_to_unique():
    """Test ToUnique functionality."""
    x = torch.randint(0, 5, (10,))
    to_unique = ToUnique()

    unique, to_orig_indices, to_csr_indices, to_csr_offsets, _ = unique_torch(x)

    assert torch.allclose(x, unique[to_orig_indices])
    assert torch.allclose(torch.sort(x[to_csr_indices]).values, x[to_csr_indices])

    unique = to_unique.to_unique(x)
    orig_x = to_unique.to_original(unique)

    assert torch.allclose(x, orig_x)
