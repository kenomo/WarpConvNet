from typing import List, Optional, Tuple

import torch
import warp as wp
from jaxtyping import Bool, Float, Int
from torch import Tensor

from warpconvnet.core.hashmap import VectorHashTable
from warpconvnet.utils.batch_index import (
    batch_index_from_offset,
    batch_indexed_coordinates,
)
from warpconvnet.utils.list_to_batch import list_to_batched_tensor
from warpconvnet.utils.ravel import ravel_mult_index_auto_shape, ravel_multi_index
from warpconvnet.utils.unique import unique_hashmap, unique_torch

__all__ = [
    "voxel_downsample_csr_mapping",
    "voxel_downsample_random_indices",
    "voxel_downsample_mapping",
    "voxel_downsample_ravel",
    "voxel_downsample_hashmap",
]


@torch.no_grad()
def voxel_downsample_hashmap(
    coords: Int[Tensor, "N D"],
):
    """
    Args:
        coords: Int[Tensor, "N D"] - coordinates

    Returns:
        unique_indices: sorted indices of unique voxels.
    """
    hash_table = VectorHashTable.from_keys(coords)
    unique_indices = hash_table.unique_index()
    return unique_indices


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
        unique_indices: sorted indices of unique voxels.
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
    # unique_indices is sorted

    if B == 1:
        batch_offsets = torch.IntTensor([0, len(unique_indices)])
    else:
        _, batch_counts = torch.unique(batch_index[unique_indices], return_counts=True)
        batch_counts = batch_counts.cpu()
        batch_offsets = torch.cat((batch_counts.new_zeros(1), batch_counts.cumsum(dim=0)))

    return unique_indices, batch_offsets


def voxel_downsample_ravel(
    batch_indexed_coords: Float[Tensor, "N D+1"],  # noqa: F821
    voxel_size: float,
):
    """
    Args:
        batch_indexed_coords: Float[Tensor, "N D+1"] - batch indexed coordinates
        voxel_size: float - voxel size

    Returns:
        unique_indices: sorted indices of unique voxels.
    """
    batch_indexed_coords[:, 1:] = torch.floor(batch_indexed_coords[:, 1:] / voxel_size).int()
    raveled_coords = ravel_mult_index_auto_shape(batch_indexed_coords)
    _, _, _, _, perm = unique_torch(raveled_coords, dim=0)
    return perm


@torch.no_grad()
def voxel_downsample_random_indices_list_of_coords(
    list_of_coords: List[Float[Tensor, "N 3"]],
    voxel_size: float,
    device: str,
) -> Tuple[Int[Tensor, "M"], Int[Tensor, "B + 1"]]:  # noqa: F821
    """
    Args:
        list_of_coords: List[Float[Tensor, "N 3"]] - list of batched coordinates
        voxel_size: float - voxel size

    Returns:
        unique_indices: sorted indices of unique voxels.
        batch_offsets: Batch offsets.
    """
    batched_coords, offsets, _ = list_to_batched_tensor(list_of_coords)
    return voxel_downsample_random_indices(batched_coords.to(device), offsets, voxel_size)


@torch.no_grad()
def voxel_downsample_mapping(
    up_batched_points: Float[Tensor, "N 3"],  # noqa: F821
    up_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    down_batched_points: Float[Tensor, "M 3"],  # noqa: F821
    down_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    voxel_size: float,
) -> Tuple[Int[Tensor, "L"], Int[Tensor, "L"], Bool[Tensor, "N"]]:  # noqa: F821
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
    return up_map, down_map, valid
