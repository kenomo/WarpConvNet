# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import NamedTuple, Optional, Tuple, Union

import torch

import cupy as cp
import math
from jaxtyping import Int
from torch import Tensor

from warpconvnet.utils.logger import get_logger
from warpconvnet.utils.argsort import argsort
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.utils.cuda_utils import load_kernel

logger = get_logger(__name__)

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


class SerializationResult(NamedTuple):
    """
    Named tuple containing serialization results.

    Attributes:
        codes: Serialization codes of the grid coordinates
        perm: Permutation that sorts coordinates by their codes (sorted_data = original_data[perm])
        inverse_perm: Inverse permutation to restore original order (original_data = sorted_data[inverse_perm])
    """

    codes: Tensor
    perm: Optional[Tensor] = None
    inverse_perm: Optional[Tensor] = None


@torch.inference_mode()
def encode(
    grid_coord: Int[Tensor, "N 3"] | Int[Tensor, "N 4"],
    batch_offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F821
    order: POINT_ORDERING | str = POINT_ORDERING.MORTON_XYZ,
    return_perm: bool = False,
    return_inverse: bool = False,
) -> Union[Int[Tensor, "N"], SerializationResult]:  # noqa: F821
    """
    Generate ordering of the grid coordinates with optional permutation and inverse permutation.

    Args:
        grid_coord: Grid coordinates (N, 3) or (N, 4)
        batch_offsets: Batch offsets for multi-batch processing.
        order: Coordinate ordering scheme (e.g., POINT_ORDERING.MORTON_XYZ, POINT_ORDERING.MORTON_XZY, etc.)
        return_perm: Whether to return the permutation that sorts the coordinates by their codes.
        return_inverse: Whether to return the inverse permutation to restore original order.

    Returns:
        If return_perm=False and return_inverse=False:
            codes: serialization codes only (backward compatibility)
        Otherwise:
            SerializationResult with codes and requested permutations

    Examples:
        ```python
        # Just get codes (backward compatibility)
        codes = encode(coords, order=POINT_ORDERING.MORTON_XYZ)

        # Get structured result with permutation
        result = encode(coords, order=POINT_ORDERING.MORTON_XYZ, return_perm=True)
        sorted_coords = coords[result.perm]

        # Get structured result with permutation and inverse (Point Transformer style)
        result = encode(coords, order=POINT_ORDERING.MORTON_XYZ,
                       return_perm=True, return_inverse=True)
        sorted_coords = coords[result.perm]
        restored_coords = sorted_coords[result.inverse_perm]  # Should equal original coords

        # Access fields
        codes = result.codes
        perm = result.perm
        inverse_perm = result.inverse_perm
        ```
    """
    if isinstance(order, str):
        order = POINT_ORDERING(order)

    # Early return for backward compatibility when no permutations requested
    if grid_coord.shape[0] == 0:
        codes = torch.empty(0, dtype=torch.int64)
    elif order in POINT_ORDERING_TO_MORTON_PERMUTATIONS.keys():
        codes = morton_code(grid_coord, batch_offsets=batch_offsets, order=order)
    elif order == POINT_ORDERING.RANDOM:
        codes = torch.randperm(grid_coord.shape[0])
    else:
        raise NotImplementedError(f"Order '{order}' not supported at the moment")

    # Early return
    if not return_perm and not return_inverse:
        return codes

    # Handle empty grid for structured result
    if (return_perm or return_inverse) and codes.shape[0] == 0:
        empty_tensor = torch.empty(0, dtype=torch.int64)
        return SerializationResult(
            codes=codes,
            perm=empty_tensor if return_perm else None,
            inverse_perm=empty_tensor if return_inverse else None,
        )

    # Generate permutation (when either return_perm or return_inverse is True)
    perm = torch.argsort(codes)

    # Generate inverse permutation if requested
    inverse_perm = None
    if return_inverse:
        inverse_perm = torch.zeros_like(perm).scatter_(
            0, perm, torch.arange(len(perm), device=perm.device)
        )

    return SerializationResult(
        codes=codes,
        perm=perm,
        inverse_perm=inverse_perm,
    )


@torch.inference_mode()
def morton_code(
    coords: Int[Tensor, "N 3"] | Int[Tensor, "N 4"],  # noqa: F821
    batch_offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F821
    threads_per_block: int = 256,
    order: POINT_ORDERING | str = POINT_ORDERING.MORTON_XYZ,
) -> Int[Tensor, "N"]:  # noqa: F821
    """
    Generate Morton codes for the input coordinates.

    Args:
        coords: Input coordinates (N, 3) or (N, 4)
        batch_offsets: Batch offsets for multi-batch processing.
        threads_per_block: CUDA threads per block
        order: Coordinate ordering scheme (e.g., POINT_ORDERING.MORTON_XYZ, POINT_ORDERING.MORTON_XZY, etc.)

    Returns:
        Morton codes

    The coords must be in the range [0, 2^16 - 1] for 16-bit path (batched)
    or effectively [0, 2^20 - 1] for 20-bit path (single batch, after normalization)
    and the result will be the z-order number of the point.
    """
    if isinstance(order, str):
        order = POINT_ORDERING(order)

    # Assert that the order is morton
    assert (
        order in POINT_ORDERING_TO_MORTON_PERMUTATIONS.keys()
    ), f"Order '{order}' not supported for morton code"

    # Empty grid handling
    if coords.shape[0] == 0:
        return torch.empty(0, dtype=torch.int64)

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
        assert bcoords.shape[1] == 4, "bcoords must be [N, 4]"

        # Ensure bcoords_cp is C-contiguous
        bcoords_cp = cp.from_dlpack(bcoords.contiguous())

        # Max coord for 16-bit is 2^16 - 1 = 65535
        coord_max = bcoords.max().item()
        if coord_max > 65535:
            logger.warning(
                f"bcoords max is {coord_max}, which is greater than 65535, which is the max value for 16-bit morton code. "
                "Truncating the coordinates to 16-bit."
            )
            div = math.ceil(coord_max / 65535)
            bcoords_cp = bcoords_cp // div

        # Kernel expects: const int* bcoords_data, int num_points, int64_t* result_order
        blocks_per_grid = math.ceil(num_points / threads_per_block)
        _assign_order_16bit_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (bcoords_cp, num_points, result_code_cp),
        )

    # Convert result from CuPy array back to PyTorch tensor on the original device
    result_code = torch.as_tensor(result_code_cp, device=device)
    return result_code
