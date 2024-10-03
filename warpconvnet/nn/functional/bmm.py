import torch
from jaxtyping import Float
from torch import Tensor

from warpconvnet.geometry.base_geometry import (
    CatBatchedFeatures,
    PadBatchedFeatures,
    SpatialFeatures,
)
from warpconvnet.ops.batch_copy import cat_to_pad, pad_to_cat


def bmm(
    sf: SpatialFeatures,
    weights: Float[Tensor, "B C_in C_out"],
) -> SpatialFeatures:
    """
    Batch matrix multiplication.
    """
    assert sf.batch_size == weights.shape[0]
    if isinstance(sf.batched_features, CatBatchedFeatures):
        bat_features = cat_to_pad(sf.feature_tensor, sf.offsets)  # BxNxC_in
        out_bat_features = torch.bmm(bat_features, weights)
        out_features = pad_to_cat(out_bat_features, sf.offsets)
        out_features = CatBatchedFeatures(out_features, sf.offsets)
    elif isinstance(sf.batched_features, PadBatchedFeatures):
        bat_features = sf.feature_tensor  # BxMxC_in
        out_bat_features = torch.bmm(bat_features, weights)  # BxMxC_out
        out_features = PadBatchedFeatures(out_bat_features, sf.offsets)
    else:
        raise ValueError(f"Unsupported batched features type: {type(sf.batched_features)}")
    return sf.replace(
        batched_features=out_features,
    )
