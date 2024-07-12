from typing import Tuple

import torch
import torch.nn as nn

from warp.convnet.geometry.base_geometry import BatchedFeatures
from warp.convnet.geometry.point_collection import PointCollection
from warp.convnet.nn.mlp import MLPBlock

__all__ = [
    "PointCollectionTransform",
    "PointCollectionCat",
    "PointCollectionSum",
    "PointCollectionMLP",
]


class PointCollectionTransform(nn.Module):
    """
    Point transform module that applies a feature transform to the input point collection.
    No spatial operations are performed.

    Hydra config example usage:

    .. code-block:: yaml

            model:
            feature_transform:
                _target_: warp.convnet.nn.point_transform.PointCollectionTransform
                feature_transform_fn: _target_: torch.nn.ReLU
    """

    def __init__(self, feature_transform_fn: nn.Module):
        super().__init__()
        self.feature_transform_fn = feature_transform_fn

    def forward(self, *pcs: Tuple[PointCollection] | PointCollection) -> PointCollection:
        """
        Apply the feature transform to the input point collection

        Args:
            pc: Input point collection

        Returns:
            Transformed point collection
        """
        assert [isinstance(pc, PointCollection) for pc in pcs] == [True] * len(pcs)
        pc = pcs[0]
        features = [pc.feature_tensor for pc in pcs]

        out_features = self.feature_transform_fn(*features)
        return PointCollection(
            batched_coordinates=pc.batched_coordinates,
            batched_features=BatchedFeatures(out_features, pc.batched_features.offsets),
        )


class PointCollectionCat(PointCollectionTransform):
    def __init__(self):
        # concatenation
        super().__init__(lambda *x: torch.concatenate(x, dim=-1))


class PointCollectionSum(PointCollectionTransform):
    def __init__(self):
        # sum
        super().__init__(lambda *x: torch.sum(torch.stack(x), dim=0))


class PointCollectionMLP(PointCollectionTransform):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__(MLPBlock(in_channels, hidden_channels, out_channels))
