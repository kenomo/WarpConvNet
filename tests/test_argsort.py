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
