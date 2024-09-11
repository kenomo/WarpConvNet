from typing import Optional

import torch.nn as nn

from warpconvnet.nn.functional.encodings import sinusoidal_encoding


class SinusoidalEncoding(nn.Module):
    """SinusoidalEncoding."""

    def __init__(
        self, num_channels: int, data_range: float = 2.0, encoding_dim: Optional[int] = None
    ):
        """
        Args:
            num_channels: Number of channels in the input data per channel. e.g. if input has 3 channels, and num_channels is 12, then there will be 12 * 3 = 36 channels in the output for each channel in the input.
            data_range: Range of the input data. max - min. e.g. if data is between 0 and 1, then data_range is 1.
            encoding_dim: Dimension to apply the encoding to. If None, the encoding is applied to the last dimension.
        """
        super().__init__()
        assert num_channels % 2 == 0, f"num_channels must be even for sin/cos, got {num_channels}"
        self.num_channels = num_channels
        self.data_range = data_range
        self.encoding_dim = encoding_dim

    def forward(self, x):
        return sinusoidal_encoding(x, self.num_channels, self.data_range, self.encoding_dim)
