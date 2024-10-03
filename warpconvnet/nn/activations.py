from typing import Union

from torch import nn

from warpconvnet.geometry.base_geometry import SpatialFeatures
from warpconvnet.nn.base_module import BaseSpatialModule
from warpconvnet.nn.functional.transforms import (
    apply_feature_transform,
    elu,
    gelu,
    leaky_relu,
    log_softmax,
    relu,
    sigmoid,
    silu,
    softmax,
    tanh,
)

__all__ = [
    "ReLU",
    "GELU",
    "SiLU",
    "Tanh",
    "Sigmoid",
    "LeakyReLU",
    "ELU",
    "Softmax",
    "LogSoftmax",
]


class ReLU(BaseSpatialModule):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def __repr__(self):
        return f"{self.__class__.__name__}(inplace={self.relu.inplace})"

    def forward(self, input: SpatialFeatures):  # noqa: F821
        return apply_feature_transform(input, self.relu)


class GELU(BaseSpatialModule):
    def forward(self, input: SpatialFeatures):  # noqa: F821
        return gelu(input)


class SiLU(BaseSpatialModule):
    def forward(self, input: SpatialFeatures):  # noqa: F821
        return silu(input)


class Tanh(BaseSpatialModule):
    def forward(self, input: SpatialFeatures):  # noqa: F821
        return tanh(input)


class Sigmoid(BaseSpatialModule):
    def forward(self, input: SpatialFeatures):  # noqa: F821
        return sigmoid(input)


class LeakyReLU(nn.Module):
    def forward(self, input: SpatialFeatures):  # noqa: F821
        return leaky_relu(input)


class ELU(BaseSpatialModule):
    def forward(self, input: SpatialFeatures):  # noqa: F821
        return elu(input)


class Softmax(BaseSpatialModule):
    def forward(self, input: SpatialFeatures):  # noqa: F821
        return softmax(input)


class LogSoftmax(BaseSpatialModule):
    def forward(self, input: SpatialFeatures):  # noqa: F821
        return log_softmax(input)
