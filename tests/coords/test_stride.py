# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.coords.ops.stride import stride_coords
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates


@pytest.fixture
def setup_voxels():
    """Setup test voxels with random coordinates."""
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, min_N, max_N, C = 3, 100000, 1000000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    voxel_size = 0.01
    coords = [(torch.rand((N, 3)) / voxel_size).int() for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    voxels = Voxels(coords, features, device=device.type).unique()
    return voxels


def test_stride_coords(setup_voxels):
    """Test basic striding functionality."""
    voxels = setup_voxels
    batch_indexed_coords = batch_indexed_coordinates(
        voxels.coordinate_tensor,
        voxels.offsets,
    )

    output_coords, offsets = stride_coords(batch_indexed_coords, stride=(2, 2, 2))

    # Test output properties
    assert output_coords.shape[0] < batch_indexed_coords.shape[0]
    assert offsets.shape == (voxels.batch_size + 1,)
