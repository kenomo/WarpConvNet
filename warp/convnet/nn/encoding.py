import torch
from torch import nn, Tensor
import numpy as np

from jaxtyping import Float


def sinusoidal_encode(
    x: Float[Tensor, "N D"], freqs: Float[Tensor, "F"]
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
        assert (
            num_channels % 2 == 0
        ), f"num_channels must be even for sin/cos, got {num_channels}"
        self.num_channels = num_channels
        self.freqs = (
            2 ** torch.arange(start=0, end=self.num_channels // 2, dtype=float)
        ) / data_range

    def __repr__(self):
        return f"SinusoidalEncoding(num_channels={self.num_channels})"

    def forward(self, x):
        return sinusoidal_encode(x, self.freqs)
