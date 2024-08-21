from enum import Enum
from typing import Tuple

from jaxtyping import Float, Int
from torch import Tensor

import warp as wp
from warp.convnet.utils.argsort import argsort
from warp.convnet.utils.batch_index import batch_indexed_coordinates


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
def morton_code(bcoord: wp.vec4i) -> wp.int64:
    """
    Assume that the coords are in the range [-2^15, 2^15 - 1] and the result_order will be the z-order number of the point.
    the batch size should be less than 2^16=65536
    """
    offset = 1 << 15
    ux = wp.int64(bcoord[1] + offset)
    uy = wp.int64(bcoord[2] + offset)
    uz = wp.int64(bcoord[3] + offset)

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
    bcoords: wp.array(dtype=wp.vec4i),
    result_order: wp.array(dtype=wp.int64),
) -> None:
    """
    Assume that the coords are in the range [-2^15, 2^15 - 1] and the result_order will be the z-order number of the point.
    the batch size should be less than 2^16=65536
    """
    tid = wp.tid()
    result_order[tid] = morton_code(bcoords[tid])


@wp.kernel
def _assign_order_discrete_20bit(
    coords: wp.array(dtype=wp.vec3i),
    result_order: wp.array(dtype=wp.int64),
) -> None:
    tid = wp.tid()
    coord = coords[tid]

    offset = 1 << 20  # Large enough to handle negative values up to 2^20
    ux = wp.int64(coord[0] + offset)
    uy = wp.int64(coord[1] + offset)
    uz = wp.int64(coord[2] + offset)

    # Calculate the Morton order
    result_order[tid] = (
        (_part1by2_long(uz) << wp.int64(2))
        | (_part1by2_long(uy) << wp.int64(1))
        | _part1by2_long(ux)
    )


def sorting_permutation(
    coords: Int[Tensor, "N 3"],  # noqa: F821
    offsets: Int[Tensor, "B"] = None,  # noqa: F821
    ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
) -> Tuple[Int[Tensor, "N"], Int[Tensor, "N"]]:  # noqa: F821
    """
    Returns the permutation of the input coordinates that sorts them according to the ordering.

    The coords must be in the range [-2^15, 2^15 - 1] and the result_order will be the z-order number of the point.
    """
    if ordering != POINT_ORDERING.Z_ORDER:
        raise NotImplementedError(f"Ordering {ordering} not implemented")

    device = str(coords.device)
    wp_result_rank = wp.zeros(shape=len(coords), dtype=wp.int64, device=device)
    if offsets is None or len(offsets) < 2:
        # Single batch
        wp_coords = wp.from_torch(coords, dtype=wp.vec3i)
        wp.launch(
            _assign_order_discrete_20bit,
            len(coords),
            inputs=[wp_coords, wp_result_rank],
            device=device,
        )
    else:
        # Multiple batches
        bcoords = coords
        if bcoords.shape[1] == 3:
            bcoords = batch_indexed_coordinates(bcoords, offsets)
        wp_coords = wp.from_torch(bcoords, dtype=wp.vec4i)
        wp.launch(
            _assign_order_discrete_16bit,
            len(bcoords),
            inputs=[wp_coords, wp_result_rank],
            device=device,
        )

    # Sort withint each offsets.
    result_rank = wp.to_torch(wp_result_rank)
    perm = argsort(result_rank)
    return perm, result_rank


def sort_point_collection(
    coords: Float[Tensor, "N 3"] | Int[Tensor, "N 3"],  # noqa: F821
    features: Float[Tensor, "N C"],  # noqa: F821
    ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
    offsets: Int[Tensor, "B"] = None,  # noqa: F821
) -> Tuple[Float[Tensor, "N 3"], Float[Tensor, "N C"]]:  # noqa: F821
    """
    Sort the point collection features using Z-ordering. The offsets remain the
    same as the permutation only sorts within each batch
    """
    sorted_order = sorting_permutation(coords, offsets, ordering)
    sorted_order = sorted_order.long()
    return coords[sorted_order], features[sorted_order]
