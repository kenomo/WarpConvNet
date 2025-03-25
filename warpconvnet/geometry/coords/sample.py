# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
from jaxtyping import Int

import torch
from torch import Tensor


def random_downsample(
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    sample_points: int,
) -> Tuple[Int[Tensor, "M"], Int[Tensor, "B+1"]]:  # noqa: F821
    """
    Randomly downsample the coordinates to the specified number of points
    """
    num_points = offsets.diff()
    batch_size = len(num_points)
    # sample sample_points per batch. BxN
    sampled_indices = torch.floor(
        torch.rand(batch_size, sample_points) * num_points.view(-1, 1)
    ).to(torch.int32)
    # Add offsets
    sampled_indices = sampled_indices + offsets[:-1].view(-1, 1)
    sampled_indices = sampled_indices.view(-1)
    return sampled_indices, offsets
