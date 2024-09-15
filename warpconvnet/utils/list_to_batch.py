from typing import List, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.utils.batch_index import batch_index_from_offset


def list_to_batched_tensor(
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


def list_to_batch_indexed_tensor(
    tensor_list: List[Float[Tensor, "N C"]],  # noqa: F821
) -> Float[Tensor, "M C+1"]:  # noqa: F821
    """
    Convert a list of tensors to a batched tensor.
    """
    batched_tensor, offsets, batch_size = list_to_batched_tensor(tensor_list)
    batch_index = batch_index_from_offset(offsets).view(-1, 1)
    batched_tensor = torch.cat([batch_index, batched_tensor], dim=-1)
    return batched_tensor
