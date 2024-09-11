import warnings
from typing import Literal, Union

import torch
from jaxtyping import Int
from torch import Tensor

import warp as wp


@wp.kernel
def prepare_key_value_pairs(
    data: wp.array(dtype=int), keys: wp.array(dtype=int), values: wp.array(dtype=int)
):
    tid = wp.tid()
    keys[tid] = data[tid]
    values[tid] = tid


def argsort(
    data: Union[wp.array, Int[Tensor, "N"]],  # noqa: F821
    backend: Literal["torch", "warp"] = "torch",
) -> Union[wp.array, Int[Tensor, "N"]]:  # noqa: F821
    """
    Sorts the input data and returns the indices that would sort the data.
    The output type will be determined by the sorting backend method.

    Args:
        data: The input data to be sorted
        backend: The backend to use for sorting
    """
    if backend == "torch":
        if isinstance(data, wp.array):
            data = wp.to_torch(data)
        return torch.argsort(data)

    warnings.warn("Using Warp argsort. warp backend could be slower than torch", stacklevel=2)

    N = len(data)
    device = str(data.device)
    keys, values = wp.empty(2 * N, dtype=int, device=device), wp.empty(
        2 * N, dtype=int, device=device
    )

    # Prepare key-value pairs
    if isinstance(data, Tensor):
        data = wp.from_torch(data.int())
    wp.launch(kernel=prepare_key_value_pairs, dim=N, inputs=[data, keys, values], device=device)

    # sort the key-value pairs
    wp.utils.radix_sort_pairs(keys, values, count=N)
    return values[:N]
