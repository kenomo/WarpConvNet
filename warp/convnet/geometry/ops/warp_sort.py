# Sort the point collection features using Z-ordering
from typing import Tuple

from enum import Enum
from jaxtyping import Float, Int
from torch import Tensor

import warp as wp


class POINT_ORDERING(Enum):
    RANDOM = Enum.auto()
    Z_ORDER = Enum.auto()


@wp.kernel
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
    dim: int = 1024,
):
    tid = wp.tid()
    # Assign result_order the z-order number of the point. Assume the points are in the range [0, 1]
    point: wp.vec3 = points[tid]
    x = point[0]
    y = point[1]
    z = point[2]
    ux = wp.clamp(int(x * dim), 0, dim - 1)
    uy = wp.clamp(int(y * dim), 0, dim - 1)
    uz = wp.clamp(int(z * dim), 0, dim - 1)
    result_order[tid] = (_part1by2(uz) << 2) | (_part1by2(uy) << 1) | _part1by2(ux)


def assign_order(
    coords: Float[Tensor, "N 3"],
    ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
    dim: int = 1024,
) -> Int[Tensor, "N"]:
    """
    Sort the points according to the ordering provided.

    The coords must be in the range [0, 1] and the result_order will be the z-order number of the point.
    """
    if ordering == POINT_ORDERING.Z_ORDER:
        result_order = wp.zeros(shape=len(coords), dtype=wp.int32)
        wp.launch(_assign_order, len(coords), inputs=[coords, result_order, dim])
        return result_order
    else:
        raise NotImplementedError(f"Ordering {ordering} not implemented")


def sort_point_collection(
    coords: Float[Tensor, "N 3"],
    features: Float[Tensor, "N C"],
    ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
    dim: int = 1024,
    offsets: Int[Tensor, "B"] = None,
) -> Tuple[Float[Tensor, "N 3"], Float[Tensor, "N C"]]:
    """
    Sort the point collection features using Z-ordering
    """
    order = assign_order(coords, ordering, dim)
    if offsets is not None:
        sorted_order = wp.argsort(order, offsets)
    else:
        # Sort within each batch
        # TODO(cchoy) accelerate for loop
        sorted_order = []
        for i in range(len(offsets) - 1):
            argsort = wp.argsort(order[offsets[i] : offsets[i + 1]])
            sorted_order.append(argsort + offsets[i])
        sorted_order = wp.concatenate(sorted_order)
    # Convert to torch tensor
    sorted_order = wp.to_torch(sorted_order).long()
    return coords[sorted_order], features[sorted_order]
