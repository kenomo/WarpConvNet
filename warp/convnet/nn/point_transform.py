import torch.nn as nn

from warp.convnet.geometry.base_geometry import BatchedFeatures
from warp.convnet.geometry.point_collection import PointCollection

__all__ = ["PointCollectionTransform"]


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

    def forward(self, pc: PointCollection) -> PointCollection:
        """
        Apply the feature transform to the input point collection

        Args:
            pc: Input point collection

        Returns:
            Transformed point collection
        """
        out_features = self.feature_transform_fn(pc.feature_tensor)
        return PointCollection(
            batched_coordinates=pc.batched_coordinates,
            batched_features=BatchedFeatures(out_features, pc.batched_features.offsets),
        )
