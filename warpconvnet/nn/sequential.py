import torch
import torch.nn as nn

from warpconvnet.geometry.base_geometry import BatchedSpatialFeatures
from warpconvnet.nn.base_module import BaseSpatialModule


class Sequential(nn.Sequential, BaseSpatialModule):
    """
    Sequential module that allows for spatial and non-spatial layers to be chained together.

    If the module has multiple consecutive non-spatial layers, then it will not create an intermediate
    spatial features object and will become more efficient.
    """

    def forward(self, x: BatchedSpatialFeatures):
        assert isinstance(
            x, BatchedSpatialFeatures
        ), f"Expected BatchedSpatialFeatures, got {type(x)}"

        in_sf = x
        for module in self:
            if isinstance(module, BaseSpatialModule) and isinstance(x, BatchedSpatialFeatures):
                x = module(x)
            elif not isinstance(module, BaseSpatialModule) and isinstance(
                x, BatchedSpatialFeatures
            ):
                in_sf = x
                x = module(x.features)
            elif isinstance(x, torch.Tensor) and isinstance(module, BaseSpatialModule):
                x = in_sf.replace(batched_features=x)
                x = module(x)
            else:
                x = module(x)

        if isinstance(x, torch.Tensor):
            x = in_sf.replace(batched_features=x)

        return x
