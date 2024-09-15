from typing import List

from torch import nn

from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.nn.functional.transforms import apply_feature_transform

__all__ = [
    "NormalizationBase",
    "BatchNorm",
    "LayerNorm",
    "InstanceNorm",
    "GroupNorm",
]


class NormalizationBase(nn.Module):
    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm

    def __repr__(self):
        return f"{self.__class__.__name__}({self.norm})"

    def forward(
        self,
        input: SpatiallySparseTensor | PointCollection,
    ):
        return apply_feature_transform(input, self.norm)


class BatchNorm(NormalizationBase):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__(nn.BatchNorm1d(num_features, eps=eps))


class LayerNorm(NormalizationBase):
    def __init__(self, normalized_shape: List[int], eps: float = 1e-5):
        super().__init__(nn.LayerNorm(normalized_shape, eps=eps))


class InstanceNorm(NormalizationBase):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__(nn.InstanceNorm1d(num_features, eps=eps))


class GroupNorm(NormalizationBase):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__(nn.GroupNorm(num_groups, num_channels, eps=eps))
