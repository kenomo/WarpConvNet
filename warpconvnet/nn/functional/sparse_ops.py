from typing import Sequence

import torch
import warp as wp

from warpconvnet.geometry.spatially_sparse_tensor import (
    BatchedFeatures,
    SpatiallySparseTensor,
)


def cat_spatially_sparse_tensors(
    *sparse_tensors: Sequence[SpatiallySparseTensor],
) -> SpatiallySparseTensor:
    """
    Concatenate a list of spatially sparse tensors.
    """
    # Check that all sparse tensors have the same offsets
    offsets = sparse_tensors[0].offsets
    for sparse_tensor in sparse_tensors:
        if not torch.allclose(sparse_tensor.offsets.to(offsets), offsets):
            raise ValueError("All sparse tensors must have the same offsets")

    # Concatenate the features tensors
    features_tensor = torch.cat(
        [sparse_tensor.feature_tensor for sparse_tensor in sparse_tensors], dim=-1
    )
    return sparse_tensors[0].replace(batched_features=BatchedFeatures(features_tensor, offsets))
