from typing import Tuple

import torch
from jaxtyping import Int
from torch import Tensor

import warp as wp
from warp.convnet.core.hashmap import VectorHashTable


def unique_torch(
    x: Int[Tensor, "N C"], dim: int = 0, stable: bool = False
) -> Tuple[  # noqa: F821
    Int[Tensor, "M C"],  # noqa: F821
    Int[Tensor, "N"],  # noqa: F821
    Int[Tensor, "N"],  # noqa: F821
    Int[Tensor, "M+1"],  # noqa: F821
    Int[Tensor, "M"],  # noqa: F821
]:
    """
    Get unique elements along a dimension.

    Args:
        x: Tensor
        dim: int
        stable: bool

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    from https://github.com/pytorch/pytorch/issues/36748
    """
    unique, to_orig_indices, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    all_to_unique_indices = to_orig_indices.argsort(stable=stable)
    all_to_unique_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))

    dtype_ind, device = to_orig_indices.dtype, to_orig_indices.device
    perm = torch.arange(x.size(dim), dtype=dtype_ind, device=device)
    perm = torch.empty(unique.size(dim), dtype=dtype_ind, device=device).scatter_(
        dim, to_orig_indices, perm
    )

    return (
        unique,
        to_orig_indices,
        all_to_unique_indices,
        all_to_unique_offsets,
        perm,
    )


def unique_hashmap(
    bcoords: Int[Tensor, "N 4"]  # noqa: F821
) -> Tuple[Int[Tensor, "M"], VectorHashTable]:  # noqa: F821
    # Append batch index to the coordinates
    assert "cuda" in str(
        bcoords.device
    ), f"Batched coordinates must be on cuda device, got {bcoords.device}"
    table = VectorHashTable(2 * len(bcoords))
    table.insert(wp.from_torch(bcoords, dtype=wp.vec4i))
    return table.unique_index(), table  # this is a torch tensor
