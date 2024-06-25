# Sort the point collection features using Z-ordering
from enum import Enum
from typing import Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

import warp as wp
from warp.convnet.utils.argsort import argsort


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
def _assign_order_continuous(
    points: wp.array(dtype=wp.vec3f),
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


@wp.kernel
def _assign_order_discrete(
    coords: wp.array(dtype=wp.vec3i),
    result_order: wp.array(dtype=wp.int32),
    min_coord: wp.vec3i,
    grid_size: int = 1024,
) -> None:
    tid = wp.tid()
    # Assign result_order the z-order number of the point. Assume the points are in the range [0, 1]
    coord = coords[tid]
    ux = wp.clamp(coord[0] - min_coord[0], 0, grid_size - 1)
    uy = wp.clamp(coord[1] - min_coord[1], 0, grid_size - 1)
    uz = wp.clamp(coord[2] - min_coord[2], 0, grid_size - 1)
    result_order[tid] = (_part1by2(uz) << 2) | (_part1by2(uy) << 1) | _part1by2(ux)


def sort_permutation(
    coords: Float[Tensor, "N 3"] | Int[Tensor, "N 3"],  # noqa: F821
    offsets: Int[Tensor, "B"] = None,  # noqa: F821
    ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
    grid_size: int = 1024,
) -> Int[Tensor, "N"]:  # noqa: F821
    """
    Sort the points according to the ordering provided.

    The coords must be in the range [0, 1] and the result_order will be the z-order number of the point.
    """
    device = str(coords.device)
    if ordering == POINT_ORDERING.Z_ORDER:
        wp_result_rank = wp.zeros(shape=len(coords), dtype=wp.int32, device=device)
        # Convert torch coords to wp.array of vec3
        if coords.dtype == torch.int32:
            wp_coords = wp.from_torch(coords, dtype=wp.vec3i)
            min_coord = wp_coords.min(dim=0)[0]
            wp.launch(
                _assign_order_discrete,
                len(coords),
                inputs=[wp_coords, wp_result_rank, min_coord, grid_size],
            )
        elif coords.dtype == torch.float32:
            wp_coords = wp.from_torch(coords, dtype=wp.vec3f)
            wp.launch(
                _assign_order_continuous,
                len(coords),
                inputs=[wp_coords, wp_result_rank, grid_size],
            )
        else:
            raise NotImplementedError(f"Coords dtype {coords.dtype} not implemented")

    else:
        raise NotImplementedError(f"Ordering {ordering} not implemented")

    # Sort withint each offsets.
    perm = argsort(wp_result_rank, device)
    return wp.to_torch(perm)


def sort_point_collection(
    coords: Float[Tensor, "N 3"] | Int[Tensor, "N 3"],  # noqa: F821
    features: Float[Tensor, "N C"],  # noqa: F821
    ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
    grid_size: int = 1024,
    offsets: Int[Tensor, "B"] = None,  # noqa: F821
) -> Tuple[Float[Tensor, "N 3"], Float[Tensor, "N C"]]:  # noqa: F821
    """
    Sort the point collection features using Z-ordering
    """
    if offsets is None:
        # Torch sort
        sorted_order = sort_permutation(coords, offsets, ordering, grid_size)
    else:
        # Sort within each batch
        sorted_order = []
        # TODO(cchoy) accelerate for loop
        for i in range(len(offsets) - 1):
            argsort = torch.argsort(rank[offsets[i] : offsets[i + 1]])  # noqa: F821
            sorted_order.append(argsort + offsets[i])
        sorted_order = torch.cat(sorted_order)

    sorted_order = sorted_order.long()
    return coords[sorted_order], features[sorted_order]
