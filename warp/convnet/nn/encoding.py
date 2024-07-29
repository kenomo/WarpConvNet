from typing import Optional

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from warp.convnet.geometry.ops.coord_ops import relative_coords
from warp.convnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult


def sinusoidal_encode(
    x: Float[Tensor, "N D"], freqs: Float[Tensor, "F"]  # noqa: F821
) -> Float[Tensor, "N 2*F*D"]:
    """
    Encode the input tensor with sinusoidal encoding.

    Args:
        x: Input tensor.
        freqs: Frequencies.

    Returns:
        Encoded tensor.
    """
    x = x.unsqueeze(-1)
    freqs = freqs.to(device=x.device, dtype=x.dtype)
    # Make freq to have the same dimensions as x. X can be of any shape
    freqs = freqs.reshape((1,) * (len(x.shape) - 1) + freqs.shape)
    # sin/cos(2 * pi * f)
    x = x * (2 * np.pi * freqs)
    x = torch.cat([x.cos(), x.sin()], dim=-1).flatten(start_dim=-2)
    return x


class SinusoidalEncoding(nn.Module):
    def __init__(self, num_channels: int, data_range: float = 2.0):
        """
        Initialize a sinusoidal encoding layer.

        Args:
            num_channels: Number of channels to encode. Must be even.
            data_range: The range of the data. For example, if the data is in the range [0, 1], then data_range=1.
        """
        super().__init__()
        assert num_channels % 2 == 0, f"num_channels must be even for sin/cos, got {num_channels}"
        self.num_channels = num_channels
        self.freqs = (
            2 ** torch.arange(start=0, end=self.num_channels // 2, dtype=float)
        ) / data_range

    def __repr__(self):
        return f"SinusoidalEncoding(num_channels={self.num_channels})"

    def forward(self, x: Float[Tensor, "N D"]) -> Float[Tensor, "N 2*F*D"]:
        return sinusoidal_encode(x, self.freqs)


class RelativeCoordsEncoding(nn.Module):
    def __init__(
        self,
        use_sinusoidal: bool = True,
        num_channels: Optional[int] = None,
        data_range: Optional[float] = 2.0,
    ):
        """
        Initialize a relative coordinates sinusoidal encoding layer.

        Args:
            num_channels: Number of channels to encode. Must be even.
            data_range: The range of the data. For example, if the data is in the range [0, 1], then data_range=1.
        """
        super().__init__()
        if use_sinusoidal:
            assert (
                num_channels is not None
            ), "num_channels must be provided when using sinusoidal encoding"
            self.sinusoidal_encoding = SinusoidalEncoding(num_channels, data_range)

    def __repr__(self):
        return f"{self.__class__.__name__}(use_sinusoidal={self.sinusoidal_encoding is not None})"

    def forward(
        self,
        neighbor_coordinates: Float[Tensor, "N D"],
        neighbor_search_result: NeighborSearchResult,
        query_coordinates: Optional[Float[Tensor, "M D"]] = None,
    ) -> Float[Tensor, "X D"]:
        rel_coords = relative_coords(
            neighbor_coordinates, neighbor_search_result, query_coordinates
        )
        if self.sinusoidal_encoding is None:
            return rel_coords
        else:
            return self.sinusoidal_encoding(rel_coords)
