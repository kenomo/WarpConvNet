from typing import Optional

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


def sinusoidal_encoding(
    x: Float[Tensor, "... D"],
    num_channels: int,
    data_range: float = 2.0,
    encoding_dim: Optional[int] = None,
) -> Float[Tensor, "... D*num_channels"]:
    """
    Apply sinusoidal encoding to the input tensor.

    Args:
        x: Input tensor of any shape.
        num_channels: Number of channels in the output per input channel.
        data_range: Range of the input data (max - min).
        encoding_dim: Dimension to apply the encoding to. If None, the encoding is applied to the last dimension.

    Returns:
        Tensor with sinusoidal encoding applied.
    """
    if encoding_dim is None:
        encoding_dim = -1
    assert num_channels % 2 == 0, f"num_channels must be even for sin/cos, got {num_channels}"

    freqs = 2 ** torch.arange(start=0, end=num_channels // 2, device=x.device).to(x.dtype)
    freqs = (2 * np.pi / data_range) * freqs
    x = x.unsqueeze(encoding_dim)
    freqs = freqs.reshape((1,) * (len(x.shape) - 1) + freqs.shape)
    x = x * freqs
    return torch.cat([x.cos(), x.sin()], dim=encoding_dim).flatten(start_dim=-2)
