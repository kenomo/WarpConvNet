from typing import List

import torch
from jaxtyping import Float
from torch import Tensor, nn

from warpconvnet.geometry.base_geometry import SpatialFeatures
from warpconvnet.nn.base_module import BaseSpatialModule
from warpconvnet.nn.functional.transforms import apply_feature_transform

__all__ = [
    "NormalizationBase",
    "BatchNorm",
    "LayerNorm",
    "InstanceNorm",
    "GroupNorm",
    "RMSNorm",
]


class NormalizationBase(BaseSpatialModule):
    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm

    def __repr__(self):
        return f"{self.__class__.__name__}({self.norm})"

    def forward(
        self,
        input: SpatialFeatures,
    ):
        return apply_feature_transform(input, self.norm)


class BatchNorm(NormalizationBase):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(nn.BatchNorm1d(num_features, eps=eps, momentum=momentum))


class LayerNorm(NormalizationBase):
    def __init__(
        self,
        normalized_shape: List[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__(
            nn.LayerNorm(
                normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias
            )
        )


class InstanceNorm(NormalizationBase):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__(nn.InstanceNorm1d(num_features, eps=eps))


class GroupNorm(NormalizationBase):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__(nn.GroupNorm(num_groups, num_channels, eps=eps))


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RMSNorm(NormalizationBase):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(_RMSNorm(dim, eps))
