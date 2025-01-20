from typing import Optional, Union
from jaxtyping import Int

from torch import Tensor

from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.pad import PadFeatures


def to_batched_features(
    features: Union[CatFeatures, PadFeatures, Tensor],
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    device: Optional[str] = None,
) -> Union[CatFeatures, PadFeatures]:
    if isinstance(features, Tensor):
        if features.ndim == 2:
            return CatFeatures(features, offsets, device=device)
        elif features.ndim == 3:
            return PadFeatures(features, offsets, device=device)
        else:
            raise ValueError(f"Invalid features tensor shape {features.shape}")
    else:
        assert isinstance(
            features, (CatFeatures, PadFeatures)
        ), f"Features must be a tensor or a CatBatchedFeatures or PadBatchedFeatures, got {type(features)}"
        if device is not None:
            features = features.to(device)
        return features
