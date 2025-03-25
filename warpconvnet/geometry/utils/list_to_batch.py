# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor


def list_to_cat_tensor(
    tensor_list: List[Float[Tensor, "N C"]],  # noqa: F821
) -> Tuple[Float[Tensor, "M C"], Int[Tensor, "B+1"], int]:  # noqa: F821
    """
    Convert a list of tensors to a batched tensor.

    Args:
        tensor_list: List of tensors to batch

    Returns:
        A tuple of the batched tensor, offsets, and batch size
    """
    offsets = [0] + [len(c) for c in tensor_list]
    # cumsum the offsets
    offsets = torch.tensor(offsets, requires_grad=False).cumsum(dim=0).int()
    batched_tensor = torch.cat(tensor_list, dim=0)
    return batched_tensor, offsets, len(offsets) - 1


def list_to_pad_tensor(
    tensor_list: List[Float[Tensor, "N C"]],  # noqa: F821
    pad_to_multiple: Optional[int] = None,
) -> Tuple[Float[Tensor, "M C"], Int[Tensor, "B+1"], int]:  # noqa: F821
    """
    Convert a list of tensors to a batched tensor.
    """
    num_points = [t.shape[0] for t in tensor_list]
    max_num_points = max(num_points)
    if pad_to_multiple is not None:
        max_num_points = (
            (max_num_points + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple
        )
    batched_tensor = torch.zeros(
        (len(tensor_list), max_num_points, tensor_list[0].shape[1]),
        dtype=tensor_list[0].dtype,
        device=tensor_list[0].device,
    )
    for i, t in enumerate(tensor_list):
        batched_tensor[i, : t.shape[0]] = t
    offsets = torch.tensor(num_points, requires_grad=False).cumsum(dim=0).int()
    offsets = torch.cat([torch.tensor([0], dtype=offsets.dtype), offsets], dim=0)
    return batched_tensor, offsets, len(offsets) - 1
