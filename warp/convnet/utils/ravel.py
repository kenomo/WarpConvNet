from typing import Tuple

import numpy as np
import torch
from jaxtyping import Int


def ravel_multi_index(
    multi_index: Int[torch.Tensor, "* D"], dims: Tuple[int, ...]  # noqa: F821
) -> Int[torch.Tensor, "*"]:
    """
    Converts a tuple of index arrays into an array of flat indices.

    Args:
        multi_index: A tensor of coordinate vectors, (*, D).
        dims: The source shape.
    """
    assert multi_index.shape[-1] == len(dims)
    # Convert dims to a list of tuples
    if isinstance(dims, torch.Tensor):
        dims = dims.cpu().tolist()
    strides = torch.tensor(
        [np.prod(dims[i + 1 :]) for i in range(len(dims))], dtype=torch.int32
    ).to(multi_index.device)
    return (multi_index * strides).sum(dim=-1)
