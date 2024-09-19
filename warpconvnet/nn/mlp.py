from torch import nn

from warpconvnet.geometry.base_geometry import BatchedFeatures, BatchedSpatialFeatures

__all__ = ["FeatureMLPBlock", "Linear"]


class FeatureMLPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = None,
        activation=nn.ReLU,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels)
        self.activation = activation()

    def forward(self, x):
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        # add skip connection
        out = self.activation(out + self.shortcut(x))
        return out


class Linear(nn.Linear):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(in_channels, out_channels, bias=bias)

    def forward(self, x: BatchedSpatialFeatures):
        return x.replace(batched_features=BatchedFeatures(super().forward(x.features), x.offsets))


class MLPBlock(FeatureMLPBlock):
    def __init__(self, in_channels: int, out_channels: int = None, hidden_channels: int = None):
        super().__init__(in_channels, out_channels, hidden_channels)

    def forward(self, x: BatchedSpatialFeatures):
        return x.replace(batched_features=BatchedFeatures(super().forward(x.features), x.offsets))
