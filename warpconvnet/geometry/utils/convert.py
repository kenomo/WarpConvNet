from typing import Optional, Union
import torch
from torch import Tensor

from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.pad import PadFeatures
from warpconvnet.geometry.base.features import Features
from warpconvnet.ops.batch_copy import cat_to_pad_tensor


def to_batched_features(
    features: Union[CatFeatures, PadFeatures, Tensor],
    offsets: Tensor,
    device: Optional[str] = None,
) -> Features:
    """Convert various feature formats to batched features."""
    if isinstance(features, (CatFeatures, PadFeatures)):
        if device is not None:
            return features.to(device)
        return features
    elif isinstance(features, torch.Tensor):
        if device is not None:
            features = features.to(device)
        return CatFeatures(features, offsets)
    else:
        raise ValueError(f"Unsupported features type: {type(features)}")
