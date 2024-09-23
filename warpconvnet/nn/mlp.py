from torch import nn

from warpconvnet.geometry.base_geometry import BatchedFeatures, BatchedSpatialFeatures

__all__ = ["FeatureResidualMLPBlock", "Linear"]


class FeatureResidualMLPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = None,
        activation=nn.ReLU,
        bias: bool = True,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.norm2 = nn.LayerNorm(out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels, bias=bias)
        self.activation = activation()

    def forward(self, x):
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        # add skip connection
        out = self.activation(out + self.shortcut(x))
        return out


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: BatchedSpatialFeatures):
        return x.replace(batched_features=BatchedFeatures(super().forward(x.features), x.offsets))


class LinearNormActivation(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features, bias=bias),
            nn.LayerNorm(out_features),
            nn.ReLU(),
        )

    def forward(self, x: BatchedSpatialFeatures):
        return x.replace(batched_features=BatchedFeatures(self.block(x.features), x.offsets))


class ResidualMLPBlock(FeatureResidualMLPBlock):
    def __init__(self, in_features: int, out_features: int = None, hidden_features: int = None):
        super().__init__(in_features, out_features, hidden_features)

    def forward(self, x: BatchedSpatialFeatures):
        return x.replace(batched_features=BatchedFeatures(super().forward(x.features), x.offsets))
