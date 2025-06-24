# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Optional, Tuple, Union

import torch

import cupy as cp
import math
from jaxtyping import Int
from torch import Tensor

from warpconvnet.utils.argsort import argsort
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.utils.cuda_utils import load_kernel


# cuda_utils.py automatically handles the csrc path for just filename
_assign_order_16bit_kernel = load_kernel(
    kernel_file="morton_code.cu", kernel_name="assign_order_discrete_16bit_kernel"
)
_assign_order_20bit_kernel_4points = load_kernel(
    kernel_file="morton_code.cu", kernel_name="assign_order_discrete_20bit_kernel_4points"
)


class POINT_ORDERING(Enum):
    RANDOM = 0
    MORTON_XYZ = 1
    MORTON_XZY = 2
    MORTON_YXZ = 3
    MORTON_YZX = 4
    MORTON_ZXY = 5
    MORTON_ZYX = 6


POINT_ORDERING_TO_MORTON_PERMUTATIONS = {
    POINT_ORDERING.MORTON_XYZ: [0, 1, 2],
    POINT_ORDERING.MORTON_XZY: [0, 2, 1],
    POINT_ORDERING.MORTON_YXZ: [1, 0, 2],
    POINT_ORDERING.MORTON_YZX: [1, 2, 0],
    POINT_ORDERING.MORTON_ZXY: [2, 0, 1],
    POINT_ORDERING.MORTON_ZYX: [2, 1, 0],
}


@torch.inference_mode()
def encode(
    grid_coord: Int[Tensor, "N 3"] | Int[Tensor, "N 4"],
    batch_offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F821
    order: POINT_ORDERING | str = POINT_ORDERING.MORTON_XYZ,
    return_perm: bool = False,
) -> Tuple[Int[Tensor, "N"], Int[Tensor, "N"]]:
    """
    Generate ordering of the grid coordinates.

    Args:
        grid_coord: Grid coordinates (N, 3)
        batch_offsets: Batch offsets for multi-batch processing.
        order: Coordinate ordering scheme (e.g., POINT_ORDERING.MORTON_XYZ, POINT_ORDERING.MORTON_XZY, etc.)
        return_perm: Whether to return the permutation of the grid coordinates. Use this to sort the input coordinate to the new ordering.
            ```python
            _, perm = encode(coords, order=POINT_ORDERING.MORTON_XYZ, return_perm=True)
            sorted_coords = coords[perm]
            ```

    Returns:
        codes: ordering of the grid coordinates
        perm: permutation of the grid coordinates
    """
    if isinstance(order, str):
        order = POINT_ORDERING(order)

    # Empty grid handling
    if grid_coord.shape[0] == 0:
        return torch.empty(0)

    # Run morton code if order is MORTON_*
    if order in POINT_ORDERING_TO_MORTON_PERMUTATIONS.keys():
        return morton_code(
            grid_coord,
            batch_offsets=batch_offsets,
            return_to_morton=return_perm,
            order=order,
        )
    elif order == POINT_ORDERING.RANDOM:
        code = torch.randperm(grid_coord.shape[0])
        if return_perm:
            return code, torch.argsort(code)
        return code
    else:
        raise NotImplementedError(f"Order '{order}' not supported at the moment")


@torch.inference_mode()
def morton_code(
    coords: Int[Tensor, "N 3"] | Int[Tensor, "N 4"],  # noqa: F821
    batch_offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F821
    return_to_morton: bool = True,
    threads_per_block: int = 256,
    order: POINT_ORDERING | str = POINT_ORDERING.MORTON_XYZ,
) -> Union[Int[Tensor, "N"], Tuple[Int[Tensor, "N"], Int[Tensor, "N"]]]:  # noqa: F821
    """
    Returns the permutation of the input coordinates that sorts them according to the ordering.

    Args:
        coords: Input coordinates (N, 3) or (N, 4)
        offsets: Batch offsets for multi-batch processing.
        return_to_morton: Whether to return sorting permutation
        threads_per_block: CUDA threads per block
        order: Coordinate ordering scheme (e.g., POINT_ORDERING.MORTON_XYZ, POINT_ORDERING.MORTON_XZY, etc.)

    Returns:
        Morton codes and optionally sorting permutation

    The coords must be in the range [0, 2^16 - 1] for 16-bit path (batched)
    or effectively [0, 2^20 - 1] for 20-bit path (single batch, after normalization)
    and the result_order will be the z-order number of the point.
    """
    if isinstance(order, str):
        order = POINT_ORDERING(order)

    # Assert that the order is morton
    assert (
        order in POINT_ORDERING_TO_MORTON_PERMUTATIONS.keys()
    ), f"Order '{order}' not supported for morton code"

    # Empty grid handling
    if coords.shape[0] == 0:
        if return_to_morton:
            return torch.empty(0), torch.empty(0)
        else:
            return torch.empty(0)

    min_coord = coords.min(0).values
    coords_normalized = (coords - min_coord).to(dtype=torch.int32).cuda()

    # Apply coordinate transformation based on ordering
    perm = POINT_ORDERING_TO_MORTON_PERMUTATIONS[order]
    if perm != [0, 1, 2]:  # Only apply permutation if it's not standard xyz
        if coords_normalized.shape[1] == 3:
            coords_normalized = coords_normalized[:, perm]
        elif coords_normalized.shape[1] == 4:  # batched coordinates [b, x, y, z]
            # Create permutation for batched coordinates: [b, x, y, z] -> [b, perm[0], perm[1], perm[2]]
            batch_perm = [0] + [p + 1 for p in perm]  # [0, perm[0]+1, perm[1]+1, perm[2]+1]
            coords_normalized = coords_normalized[:, batch_perm]

    device = coords_normalized.device
    num_points = len(coords_normalized)
    result_code_cp = cp.empty(num_points, dtype=cp.int64)

    if coords_normalized.shape[1] == 3 and (batch_offsets is None or len(batch_offsets) < 2):
        # Single batch path (20-bit)
        coords_cp = cp.from_dlpack(coords_normalized.contiguous())
        # The kernel loads 4 points per thread, so we need to adjust the number of blocks
        blocks_per_grid = math.ceil(num_points / (threads_per_block * 4))
        _assign_order_20bit_kernel_4points(
            (blocks_per_grid,),
            (threads_per_block,),
            (coords_cp, num_points, result_code_cp),
        )
    elif coords_normalized.shape[1] in [3, 4]:
        # Convert Nx3 or Nx4 to Nx4

        # Multiple batches path (16-bit)
        bcoords = coords_normalized
        if bcoords.shape[1] == 3 and batch_offsets is not None:
            bcoords = batch_indexed_coordinates(bcoords, batch_offsets).to(
                device=device, dtype=torch.int32
            )
        # bcoords is now [N, 4]

        # Ensure bcoords_cp is C-contiguous
        bcoords_cp = cp.from_dlpack(bcoords.contiguous())

        # Kernel expects: const int* bcoords_data, int num_points, int64_t* result_order
        blocks_per_grid = math.ceil(num_points / threads_per_block)
        _assign_order_16bit_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (bcoords_cp, num_points, result_code_cp),
        )

    # Convert result from CuPy array back to PyTorch tensor on the original device
    result_code = torch.from_dlpack(result_code_cp).to(device)

    if return_to_morton:
        to_morton_order = argsort(result_code, backend="torch")
        return result_code, to_morton_order
    return result_code
