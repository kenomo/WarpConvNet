# Sort the point collection features using Z-ordering
from typing import Tuple

from enum import Enum
from jaxtyping import Float, Int

import torch
from torch import Tensor

import warp as wp


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


@wp.kernel
def _assign_order(
    points: wp.array(dtype=wp.vec3),
    result_order: wp.array(dtype=wp.int32),
    grid_size: int = 1024,  # 10 bits for each x, y, z
) -> None:
    tid = wp.tid()
    # Assign result_order the z-order number of the point. Assume the points are in the range [0, 1]
    point = points[tid]
    f_grid_size = float(grid_size)
    x = wp.floor(point[0] * f_grid_size)
    y = wp.floor(point[1] * f_grid_size)
    z = wp.floor(point[2] * f_grid_size)
    ux = wp.clamp(int(x), 0, grid_size - 1)
    uy = wp.clamp(int(y), 0, grid_size - 1)
    uz = wp.clamp(int(z), 0, grid_size - 1)
    # 32-bit and 31-bit without the sign bit. 10 bits each
    result_order[tid] = (_part1by2(uz) << 2) | (_part1by2(uy) << 1) | _part1by2(ux)


def assign_rank(
    coords: Float[Tensor, "N 3"],
    ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
    grid_size: int = 1024,
) -> Int[Tensor, "N"]:
    """
    Sort the points according to the ordering provided.

    The coords must be in the range [0, 1] and the result_order will be the z-order number of the point.
    """
    if ordering == POINT_ORDERING.Z_ORDER:
        wp_result_rank = wp.zeros(shape=len(coords), dtype=wp.int32, device=str(coords.device))
        # Convert torch coords to wp.array of vec3
        wp_coords = wp.from_torch(coords, dtype=wp.vec3)
        wp.launch(_assign_order, len(coords), inputs=[wp_coords, wp_result_rank, grid_size])
        result_rank = wp.to_torch(wp_result_rank)
        return result_rank
    else:
        raise NotImplementedError(f"Ordering {ordering} not implemented")


def sort_point_collection(
    coords: Float[Tensor, "N 3"],
    features: Float[Tensor, "N C"],
    ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
    grid_size: int = 1024,
    offsets: Int[Tensor, "B"] = None,
) -> Tuple[Float[Tensor, "N 3"], Float[Tensor, "N C"]]:
    """
    Sort the point collection features using Z-ordering
    """
    rank = assign_rank(coords, ordering, grid_size)
    if offsets is None:
        # Torch sort
        sorted_order = torch.argsort(rank)
    else:
        # Sort within each batch
        sorted_order = []
        # TODO(cchoy) accelerate for loop
        for i in range(len(offsets) - 1):
            argsort = torch.argsort(rank[offsets[i] : offsets[i + 1]])
            sorted_order.append(argsort + offsets[i])
        sorted_order = torch.cat(sorted_order)

    sorted_order = sorted_order.long()
    return coords[sorted_order], features[sorted_order]
