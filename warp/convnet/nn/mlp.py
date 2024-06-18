from torch import nn

from warp.convnet.geometry.point_collection import BatchedFeatures, PointCollection


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = None,
        out_channels: int = None,
        activation=nn.GELU,
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


class PointCollectionTransform(nn.Module):
    def __init__(
        self,
        transform: nn.Module,
    ):
        super().__init__()
        self.transform = transform

    def forward(self, x: PointCollection):
        return PointCollection(
            batched_coordinates=x.batched_coordinates,
            batched_features=BatchedFeatures(
                self.transform(x.features),
                offsets=x.batched_features.offsets,
            ),
        )
