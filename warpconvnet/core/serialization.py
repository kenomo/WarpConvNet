from enum import Enum
from typing import Optional, Tuple

import torch
import warp as wp
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.utils.argsort import argsort
from warpconvnet.utils.batch_index import batch_indexed_coordinates


class POINT_ORDERING(Enum):
    RANDOM = 0
    Z_ORDER = 1


@wp.func
def _part1by2(n: int) -> int:
    n = (n ^ (n << 16)) & 0xFF0000FF
    n = (n ^ (n << 8)) & 0x0300F00F
    n = (n ^ (n << 4)) & 0x030C30C3
    n = (n ^ (n << 2)) & 0x09249249
    return n


@wp.func
def _part1by2_long(n: wp.int64) -> wp.int64:
    n = (n ^ (n << wp.int64(32))) & wp.int64(0xFFFF00000000FFFF)
    n = (n ^ (n << wp.int64(16))) & wp.int64(0x00FF0000FF0000FF)
    n = (n ^ (n << wp.int64(8))) & wp.int64(0xF00F00F00F00F00F)
    n = (n ^ (n << wp.int64(4))) & wp.int64(0x30C30C30C30C30C3)
    n = (n ^ (n << wp.int64(2))) & wp.int64(0x9249249249249249)
    return n


@wp.func
def _morton_code(bcoord: wp.array(dtype=int)) -> wp.int64:
    """
    Assume that the coords are in the range [0, 2^16 - 1] and the result_order will be the z-order number of the point.
    the batch size should be less than 2^15=32768
    """
    # offset = 1 << 15
    ux = wp.int64(bcoord[1])
    uy = wp.int64(bcoord[2])
    uz = wp.int64(bcoord[3])

    # Calculate the Morton order
    morton_code = (
        (_part1by2_long(uz) << wp.int64(2))
        | (_part1by2_long(uy) << wp.int64(1))
        | _part1by2_long(ux)
    )

    # Erase the first 16 bits of the Morton code to make space for the batch index
    morton_code &= wp.int64(0x0000FFFFFFFFFFFF)

    # Combine the batch index with the Morton order to ensure batch continuity
    return (wp.int64(bcoord[0]) << wp.int64(48)) | morton_code


@wp.kernel
def _assign_order_discrete_16bit(
    bcoords: wp.array2d(dtype=int),
    result_order: wp.array(dtype=wp.int64),
) -> None:
    """
    Assume that the coords are in the range [0, 2^16 - 1] and the result_order will be the z-order number of the point.
    the batch size should be less than 2^15=32768
    """
    tid = wp.tid()
    result_order[tid] = _morton_code(bcoords[tid])


@wp.kernel
def _assign_order_discrete_20bit(
    coords: wp.array(dtype=wp.vec3i),
    result_order: wp.array(dtype=wp.int64),
) -> None:
    tid = wp.tid()
    coord = coords[tid]

    # offset = 1 << 20  # Large enough to handle negative values up to 2^20
    ux = wp.int64(coord[0])
    uy = wp.int64(coord[1])
    uz = wp.int64(coord[2])

    # Calculate the Morton order
    result_order[tid] = (
        (_part1by2_long(uz) << wp.int64(2))
        | (_part1by2_long(uy) << wp.int64(1))
        | _part1by2_long(ux)
    )


@torch.inference_mode()
def morton_code(
    coords: Int[Tensor, "N 3"] | Int[Tensor, "N 4"],  # noqa: F821
    offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F821
    return_to_morton: bool = True,
) -> Tuple[Int[Tensor, "N"], Int[Tensor, "N"]]:  # noqa: F821
    """
    Returns the permutation of the input coordinates that sorts them according to the ordering.

    The coords must be in the range [0, 2^16 - 1] and the result_order will be the z-order number of the point.
    """
    min_coord = coords.min(0).values
    coords = (coords - min_coord).to(torch.int32)
    device = str(coords.device)
    wp_result_code = wp.zeros(shape=len(coords), dtype=wp.int64, device=device)
    if coords.shape[1] == 3 and (offsets is None or len(offsets) < 2):
        # Single batch
        wp_coords = wp.from_torch(coords, dtype=wp.vec3i)
        wp.launch(
            _assign_order_discrete_20bit,
            len(coords),
            inputs=[wp_coords, wp_result_code],
            device=device,
        )
    else:
        # Multiple batches
        bcoords = coords
        if bcoords.shape[1] == 3 and offsets is not None:
            bcoords = batch_indexed_coordinates(bcoords, offsets)
        wp_coords = wp.from_torch(bcoords)
        wp.launch(
            _assign_order_discrete_16bit,
            len(bcoords),
            inputs=[wp_coords, wp_result_code],
            device=device,
        )

    # Sort withint each offsets.
    result_code = wp.to_torch(wp_result_code)
    if return_to_morton:
        to_morton_order = argsort(result_code, backend="torch")
        return result_code, to_morton_order
    return result_code
