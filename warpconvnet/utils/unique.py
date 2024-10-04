from typing import Literal, Tuple

import torch
import warp as wp
from jaxtyping import Int
from torch import Tensor

from warpconvnet.core.hashmap import HashMethod, VectorHashTable
from warpconvnet.utils.ravel import ravel_multi_index


def unique_inverse(
    x: Tensor,
    dim: int = 0,
) -> Tuple[Int[Tensor, "M C"], Int[Tensor, "N"],]:  # noqa: F821  # noqa: F821  # noqa: F821
    """
    Get to_unique_indices and to_orig_indices.
    """
    unique, to_orig_indices = torch.unique(x, dim=dim, sorted=True, return_inverse=True)
    to_unique_indices = torch.arange(x.size(dim), dtype=torch.int32, device=x.device).scatter_(
        dim=0, index=to_orig_indices, src=unique
    )
    return to_unique_indices, to_orig_indices


def unique_torch(
    x: Int[Tensor, "N C"],
    dim: int = 0,
    stable: bool = False,
    return_to_unique_indices: bool = False,
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
        unique: M unique coordinates
        to_orig_indices: N indices to original coordinates. unique[to_orig_indices] == x
        all_to_csr_indices: N indices to unique coordinates. x[all_to_csr_indices] == torch.repeat_interleave(unique, counts).
        all_to_csr_offsets: M+1 offsets to unique coordinates. counts = all_to_csr_offsets.diff()
        to_unique_indices: M indices to sample x to unique. x[to_unique_indices] == unique

    from https://github.com/pytorch/pytorch/issues/36748
    """
    unique, to_orig_indices, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    all_to_csr_indices = to_orig_indices.argsort(stable=stable)
    all_to_csr_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))

    if return_to_unique_indices:
        dtype_ind, device = to_orig_indices.dtype, to_orig_indices.device
        to_unique_indices = torch.arange(x.size(dim), dtype=dtype_ind, device=device)
        to_unique_indices = torch.empty(unique.size(dim), dtype=dtype_ind, device=device).scatter_(
            dim, to_orig_indices, to_unique_indices
        )
    else:
        to_unique_indices = None

    return (
        unique,
        to_orig_indices,
        all_to_csr_indices,
        all_to_csr_offsets,
        to_unique_indices,
    )


def unique_ravel(
    x: Int[Tensor, "N C"],
    dim: int = 0,
    sorted: bool = False,
):
    min_coords = x.min(dim=dim).values
    shifted_x = x - min_coords
    shape = shifted_x.max(dim=dim).values + 1
    raveled_x = ravel_multi_index(shifted_x, shape)
    unique_raveled_x, _, _, _, perm = unique_torch(raveled_x, dim=0)
    if sorted:
        perm = perm[unique_raveled_x.argsort()]
    return perm


def unique_hashmap(
    bcoords: Int[Tensor, "N 4"],  # noqa: F821
    hash_method: HashMethod = HashMethod.CITY,
) -> Tuple[Int[Tensor, "M"], VectorHashTable]:  # noqa: F821
    """
    Args:
        bcoords: Batched coordinates.
        hash_method: Hash method.

    Returns:
        unique_indices: bcoords[unique_indices] == unique
        hash_table: Hash table.
    """
    # Append batch index to the coordinates
    assert "cuda" in str(
        bcoords.device
    ), f"Batched coordinates must be on cuda device, got {bcoords.device}"
    table = VectorHashTable(2 * len(bcoords), hash_method)
    table.insert(wp.from_torch(bcoords))
    return table.unique_index(), table  # this is a torch tensor
