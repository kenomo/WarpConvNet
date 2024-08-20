from typing import Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warp.convnet.utils.batch_index import batch_index_from_offset
from warp.convnet.utils.unique import unique_hashmap, unique_torch


# Voxel downsample
@torch.no_grad()
def voxel_downsample_csr_mapping(
    batched_points: Float[Tensor, "N 3"],  # noqa: F722,F821
    offsets: Int[Tensor, "B + 1"],  # noqa: F722,F821
    voxel_size: float,
):
    """
    Voxel downsample the coordinates

    - floor the points to the voxel coordinates
    - concat batch index to the voxel coordinates to create batched coordinates
    - hash the batched coordinates
    - get the unique hash values
    - get the unique voxel centers

    Args:
        batched_points: Float[Tensor, "N 3"] - batched points
        offsets: Int[Tensor, "B + 1"] - offsets for each batch
        voxel_size: float - voxel size

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor] - perm, unique_offsets, to_unique_index, index_offsets
    """
    # Floor the points to the voxel coordinates
    N = len(batched_points)
    B = len(offsets) - 1
    device = str(batched_points.device)
    assert offsets[-1] == N, f"Offsets {offsets} does not match the number of points {N}"

    voxel_coords = torch.floor(batched_points / voxel_size).int()
    if B > 1:
        batch_index = batch_index_from_offset(offsets, device)
        voxel_coords = torch.cat([batch_index.unsqueeze(1), voxel_coords], dim=1)

    unique_vox_coords, inverse, to_unique_index, index_offsets, perm = unique_torch(
        voxel_coords, dim=0
    )

    if B == 1:
        unique_offsets = torch.IntTensor([0, len(unique_vox_coords)])
    else:
        _, batch_counts = torch.unique(batch_index[perm], return_counts=True)
        batch_counts = batch_counts.cpu()
        unique_offsets = torch.cat((batch_counts.new_zeros(1), batch_counts.cumsum(dim=0)))
    assert len(unique_offsets) == B + 1

    return perm, unique_offsets, to_unique_index, index_offsets


@torch.no_grad()
def voxel_downsample_random_indices(
    batched_points: Float[Tensor, "N 3"],  # noqa: F821
    offsets: Int[Tensor, "B + 1"],  # noqa: F821
    voxel_size: Optional[float] = None,
) -> Tuple[Int[Tensor, "M"], Int[Tensor, "B + 1"]]:  # noqa: F821
    """
    Args:
        batched points: Float[Tensor, "N 3"] - batched points
        offsets: Int[Tensor, "B + 1"] - offsets for each batch
        voxel_size: Optional[float] - voxel size. Will quantize the points if voxel_size is provided.

    Returns:
        unique_indices: bcoords[unique_indices] will be unique.
        batch_offsets: Batch offsets.
    """

    # Floor the points to the voxel coordinates
    N = len(batched_points)
    B = len(offsets) - 1
    device = str(batched_points.device)
    assert offsets[-1] == N, f"Offsets {offsets} does not match the number of points {N}"

    if voxel_size is not None:
        voxel_coords = torch.floor(batched_points / voxel_size).int()
    else:
        voxel_coords = batched_points.int()
    batch_index = batch_index_from_offset(offsets, device=device)
    voxel_coords = torch.cat([batch_index.unsqueeze(1), voxel_coords], dim=1)

    unique_indices, hash_table = unique_hashmap(voxel_coords)

    if B == 1:
        batch_offsets = torch.IntTensor([0, len(unique_indices)])
    else:
        _, batch_counts = torch.unique(batch_index[unique_indices], return_counts=True)
        batch_counts = batch_counts.cpu()
        batch_offsets = torch.cat((batch_counts.new_zeros(1), batch_counts.cumsum(dim=0)))

    return unique_indices, batch_offsets
