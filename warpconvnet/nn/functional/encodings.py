from typing import Optional

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


def get_freqs(
    num_channels: int, data_range: float = 2.0, device: Optional[torch.device] = None
) -> Float[Tensor, "num_channels"]:  # noqa: F821
    if device is None:
        device = torch.device("cpu")
    freqs = 2 ** torch.arange(start=0, end=num_channels // 2, device=device)
    freqs = (2 * np.pi / data_range) * freqs
    return freqs


def sinusoidal_encoding(
    x: Float[Tensor, "... D"],
    num_channels: Optional[int] = None,
    data_range: Optional[float] = None,
    encoding_axis: Optional[int] = None,
    freqs: Optional[Float[Tensor, "num_channels"]] = None,  # noqa: F821
) -> Float[Tensor, "... D*num_channels"]:
    """
    Apply sinusoidal encoding to the input tensor.

    Args:
        x: Input tensor of any shape.
        num_channels: Number of channels in the output per input channel.
        data_range: Range of the input data (max - min).
        encoding_axis: Axis to apply the encoding to. If None, the encoding is applied to the last axis.
        freqs: Frequencies to use for the sinusoidal encoding. If None, the frequencies are calculated from the data range and num_channels.

    Returns:
        Tensor with sinusoidal encoding applied.
        For input shape [..., C], the output shape is [..., C*num_channels].
    """
    if encoding_axis is None:
        encoding_axis = -1

    x = x.unsqueeze(encoding_axis)
    if freqs is None:
        assert (
            num_channels is not None and data_range is not None
        ), "num_channels and data_range must be provided if freqs are not given"
        assert num_channels % 2 == 0, f"num_channels must be even for sin/cos, got {num_channels}"
        freqs = get_freqs(num_channels, data_range, device=x.device)
    freqs = freqs.reshape((1,) * (len(x.shape) - 1) + freqs.shape)
    x = x * freqs
    return torch.cat([x.cos(), x.sin()], dim=encoding_axis).flatten(start_dim=-2)
