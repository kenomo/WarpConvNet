from typing import Union

from torch import nn

from warpconvnet.nn.functional.transforms import (
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


class ReLU(nn.Module):
    def forward(self, input: Union["SpatiallySparseTensor", "PointCollection"]):  # noqa: F821
        return relu(input)


class GELU(nn.Module):
    def forward(self, input: Union["SpatiallySparseTensor", "PointCollection"]):  # noqa: F821
        return gelu(input)


class SiLU(nn.Module):
    def forward(self, input: Union["SpatiallySparseTensor", "PointCollection"]):  # noqa: F821
        return silu(input)


class Tanh(nn.Module):
    def forward(self, input: Union["SpatiallySparseTensor", "PointCollection"]):  # noqa: F821
        return tanh(input)


class Sigmoid(nn.Module):
    def forward(self, input: Union["SpatiallySparseTensor", "PointCollection"]):  # noqa: F821
        return sigmoid(input)


class LeakyReLU(nn.Module):
    def forward(self, input: Union["SpatiallySparseTensor", "PointCollection"]):  # noqa: F821
        return leaky_relu(input)


class ELU(nn.Module):
    def forward(self, input: Union["SpatiallySparseTensor", "PointCollection"]):  # noqa: F821
        return elu(input)


class Softmax(nn.Module):
    def forward(self, input: Union["SpatiallySparseTensor", "PointCollection"]):  # noqa: F821
        return softmax(input)


class LogSoftmax(nn.Module):
    def forward(self, input: Union["SpatiallySparseTensor", "PointCollection"]):  # noqa: F821
        return log_softmax(input)
