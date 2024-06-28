from typing import Literal

import torch

import warp as wp


@wp.kernel
def prepare_key_value_pairs(
    data: wp.array(dtype=int), keys: wp.array(dtype=int), values: wp.array(dtype=int)
):
    tid = wp.tid()
    keys[tid] = data[tid]
    values[tid] = tid


def argsort(
    data: wp.array(dtype=int), device: str, backend: Literal["torch", "warp"] = "warp"
) -> wp.array(dtype=int):
    if backend == "torch":
        if isinstance(data, wp.array):
            data = wp.to_torch(data)
        return torch.argsort(data)

    N = len(data)
    keys, values = wp.empty(N, dtype=int, device=device), wp.empty(N, dtype=int, device=device)

    # Prepare key-value pairs
    wp.launch(kernel=prepare_key_value_pairs, dim=N, inputs=[data, keys, values], device=device)
    wp.utils.radix_sort_pairs(keys, values)
    return values
