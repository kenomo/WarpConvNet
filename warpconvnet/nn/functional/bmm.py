import torch
from jaxtyping import Float
from torch import Tensor

from warpconvnet.geometry.base_geometry import BatchedFeatures, BatchedSpatialFeatures
from warpconvnet.ops.batch_copy import batch_to_cat, cat_to_batch


def bmm(
    sf: BatchedSpatialFeatures,
    weights: Float[Tensor, "B C_in C_out"],
) -> torch.Tensor:
    """
    Batch matrix multiplication.
    """
    assert sf.batch_size == weights.shape[0]
    bat_features = cat_to_batch(sf.features, sf.offsets)  # BxNxC_in
    out_bat_features = torch.bmm(bat_features, weights)
    out_features = batch_to_cat(out_bat_features, sf.offsets)
    return sf.replace(
        batched_features=BatchedFeatures(
            batched_tensor=out_features,
            offsets=sf.offsets,
        ),
    )
