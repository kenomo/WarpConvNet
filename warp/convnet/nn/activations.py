from torch import nn

from warp.convnet.geometry.point_collection import PointCollection
from warp.convnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warp.convnet.nn.functional.transforms import (
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


class ReLU(nn.Module):
    def forward(self, input: SpatiallySparseTensor | PointCollection):
        return relu(input)


class GELU(nn.Module):
    def forward(self, input: SpatiallySparseTensor | PointCollection):
        return gelu(input)


class SiLU(nn.Module):
    def forward(self, input: SpatiallySparseTensor | PointCollection):
        return silu(input)


class Tanh(nn.Module):
    def forward(self, input: SpatiallySparseTensor | PointCollection):
        return tanh(input)


class Sigmoid(nn.Module):
    def forward(self, input: SpatiallySparseTensor | PointCollection):
        return sigmoid(input)


class LeakyReLU(nn.Module):
    def forward(self, input: SpatiallySparseTensor | PointCollection):
        return leaky_relu(input)


class ELU(nn.Module):
    def forward(self, input: SpatiallySparseTensor | PointCollection):
        return elu(input)


class Softmax(nn.Module):
    def forward(self, input: SpatiallySparseTensor | PointCollection):
        return softmax(input)


class LogSoftmax(nn.Module):
    def forward(self, input: SpatiallySparseTensor | PointCollection):
        return log_softmax(input)
