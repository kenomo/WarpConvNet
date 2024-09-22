from typing import Optional

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from warpconvnet.geometry.ops.coord_ops import relative_coords
from warpconvnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult
from warpconvnet.nn.functional.encodings import get_freqs, sinusoidal_encoding


class SinusoidalEncoding(nn.Module):
    def __init__(self, num_channels: int, data_range: float = 2.0, concat_input: bool = True):
        """
        Initialize a sinusoidal encoding layer.

        Args:
            num_channels: Number of channels to encode. Must be even.
            data_range: The range of the data. For example, if the data is in the range [0, 1], then data_range=1.
            concat_input: Whether to concatenate the input to the output.
        """
        super().__init__()
        assert num_channels % 2 == 0, f"num_channels must be even for sin/cos, got {num_channels}"
        self.num_channels = num_channels
        self.concat_input = concat_input
        self.register_buffer("freqs", get_freqs(num_channels // 2, data_range))

    def num_output_channels(self, num_input_channels: int) -> int:
        if self.concat_input:
            return (num_input_channels + 1) * self.num_channels
        else:
            return num_input_channels * self.num_channels

    def forward(self, x: Float[Tensor, "* C"]) -> Float[Tensor, "* num_channels*C"]:  # noqa: F821
        return sinusoidal_encoding(x, freqs=self.freqs, concat_input=self.concat_input)


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
