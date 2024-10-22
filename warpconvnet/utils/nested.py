from typing import List, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.types import NestedTensor


def to_nested(
    tensor: Float[Tensor, "N"], offsets: Int[Tensor, "B + 1"]  # noqa: F821
) -> NestedTensor:
    return torch.nested.nested_tensor(
        [tensor[start:end] for start, end in zip(offsets[:-1], offsets[1:])]
    )
