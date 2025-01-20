from dataclasses import dataclass
from typing import Optional

from torch import Tensor
from jaxtyping import Float, Int

from ..base import Geometry


@dataclass
class Points(Geometry):
    """Continuous point cloud representation.

    Points are represented as continuous coordinates in space.
    No discretization or quantization is applied.

    Args:
        coords: [N, D] continuous coordinate tensor
        features: [N, C] feature tensor
        offsets: [B+1] batch offsets tensor
    """

    coords: Float[Tensor, "N D"]
    features: Float[Tensor, "N C"]
    offsets: Int[Tensor, "B+1"]  # noqa: F821
