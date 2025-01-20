from dataclasses import dataclass
from typing import Optional, Tuple

from torch import Tensor
from jaxtyping import Float, Int

from ..base import Geometry


@dataclass
class Voxels(Geometry):
    """Sparse voxel representation.

    Points are quantized to a regular grid with voxel_size spacing.
    Only occupied voxels are stored (sparse representation).

    Args:
        coords: [N, D] discrete coordinate tensor
        features: [N, C] feature tensor
        offsets: [B+1] batch offsets tensor
        voxel_size: Size of voxels for quantization
        origin: Optional origin point for voxel grid
    """

    coords: Float[Tensor, "N D"]
    features: Float[Tensor, "N C"]
    offsets: Int[Tensor, "B+1"]  # noqa: F821
    voxel_size: float
    origin: Optional[Float[Tensor, "D"]] = None  # noqa: F821
