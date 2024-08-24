from typing import Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

import warp as wp
from warp.convnet.core.hashmap import VectorHashTable
from warp.convnet.utils.batch_index import (
    batch_index_from_offset,
    batch_indexed_coordinates,
)
from warp.convnet.utils.unique import unique_hashmap, unique_torch

__all__ = [
    "voxel_downsample_csr_mapping",
    "voxel_downsample_random_indices",
    "voxel_downsample_mapping",
]


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


@torch.no_grad()
def voxel_downsample_mapping(
    up_batched_points: Float[Tensor, "N 3"],  # noqa: F821
    up_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    down_batched_points: Float[Tensor, "M 3"],  # noqa: F821
    down_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    voxel_size: float,
) -> Tuple[Int[Tensor, "L"], Int[Tensor, "L"]]:  # noqa: F821
    """
    Find the mapping that select points in the up_batched_points that are in the down_batched_points up to voxel_size.
    The mapping is random up to voxel_size. If there is a corresponding point in the down_batched_points, the mapping
    will find a point in random within a voxel_size that is in the up_batched_points.

    up_batched_points[up_map] ~= down_batched_points[down_map]
    """
    # Only support CUDA, must be on the same device
    device = str(up_batched_points.device)
    assert "cuda" in device, "voxel_downsample_mapping only supports CUDA device"
    assert device == str(
        down_batched_points.device
    ), "up_batched_points and down_batched_points must be on the same device"

    # Convert the batched points to voxel coordinates
    up_batched_points = torch.floor(up_batched_points / voxel_size).int()
    down_batched_points = torch.floor(down_batched_points / voxel_size).int()

    # Get the batch index
    wp_up_bcoords = batch_indexed_coordinates(up_batched_points, up_offsets, return_type="warp")
    wp_down_bcoords = batch_indexed_coordinates(
        down_batched_points, down_offsets, return_type="warp"
    )

    up_table = VectorHashTable.from_keys(wp_up_bcoords)
    # Get the map that maps up_batched_points[up_map] ~= down_batched_points.
    wp_up_map = up_table.search(wp_down_bcoords)
    # remove invalid mappings (i.e. i < 0)
    up_map = wp.to_torch(wp_up_map)
    valid = up_map >= 0
    up_map = up_map[valid]
    # Get the index of true values
    down_map = torch.nonzero(valid).squeeze(1)
    return up_map, down_map
