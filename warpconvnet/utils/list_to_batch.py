from typing import List, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor


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
