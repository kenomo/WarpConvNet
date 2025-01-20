from typing import Sequence

import torch

from warpconvnet.geometry.types.voxels import Voxels


def cat_spatially_sparse_tensors(
    *sparse_tensors: Sequence[Voxels],
) -> Voxels:
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
    return sparse_tensors[0].replace(batched_features=features_tensor)
