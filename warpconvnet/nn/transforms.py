from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from warpconvnet.geometry.base_geometry import BatchedFeatures, BatchedSpatialFeatures
from warpconvnet.nn.mlp import MLPBlock

__all__ = [
    "Transform",
    "Cat",
    "Sum",
    "Linear",
    "MLP",
]


class Transform(nn.Module):
    """
    Point transform module that applies a feature transform to the input point collection.
    No spatial operations are performed.

    Hydra config example usage:

    .. code-block:: yaml

            model:
            feature_transform:
                _target_: warpconvnet.nn.point_transform.Transform
                feature_transform_fn: _target_: torch.nn.ReLU
    """

    def __init__(self, feature_transform_fn: nn.Module):
        super().__init__()
        self.feature_transform_fn = feature_transform_fn

    def forward(self, *sfs: Tuple[BatchedSpatialFeatures, ...]) -> BatchedSpatialFeatures:
        """
        Apply the feature transform to the input point collection

        Args:
            pc: Input point collection

        Returns:
            Transformed point collection
        """
        assert [isinstance(sf, BatchedSpatialFeatures) for sf in sfs] == [True] * len(sfs)
        # Assert that all spatial features have the same offsets
        assert all(torch.allclose(sf.offsets, sfs[0].offsets) for sf in sfs)
        sf = sfs[0]
        features = [sf.feature_tensor for sf in sfs]

        out_features = self.feature_transform_fn(*features)
        return sf.replace(
            batched_features=BatchedFeatures(out_features, sf.offsets),
        )


class Cat(Transform):
    def __init__(self):
        # concatenation
        super().__init__(lambda *x: torch.concatenate(x, dim=-1))


class Sum(Transform):
    @staticmethod
    def _sum_fn(cls, xs: Tuple[Tensor, ...]) -> Tensor:
        feat = xs[0].clone()
        for x in xs[1:]:
            feat += x
        return feat

    def __init__(self):
        super().__init__(self._sum_fn)


class Linear(Transform):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(nn.Linear(in_channels, out_channels, bias=bias))


class MLP(Transform):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__(MLPBlock(in_channels, hidden_channels, out_channels))
