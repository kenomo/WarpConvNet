from typing import List, Tuple

import pytest
import torch
from torch import Tensor

from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.pad import PadFeatures


@pytest.fixture
def setup_cat_features() -> Tuple[List[Tensor], CatFeatures, Tensor]:
    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    features = [torch.rand((N, C)) for N in Ns]
    cat_features = CatFeatures(
        torch.cat(features, dim=0),
        torch.tensor([0] + list(torch.cumsum(Ns, dim=0).numpy())),
    )
    return features, cat_features, Ns


def test_cat_features_init(setup_cat_features):
    features, cat_features, Ns = setup_cat_features

    # Test initialization from list
    cat_from_list = CatFeatures(features)
    assert cat_from_list.num_channels == features[0].shape[1]
    assert (cat_from_list.offsets == cat_features.offsets).all()

    # Test initialization from tensor and offsets
    tensor = torch.cat(features, dim=0)
    offsets = torch.tensor([0] + list(torch.cumsum(Ns, dim=0).numpy()))
    cat_from_tensor = CatFeatures(tensor, offsets)
    assert (cat_from_tensor.batched_tensor == cat_features.batched_tensor).all()


def test_cat_features_properties(setup_cat_features):
    _, cat_features, _ = setup_cat_features

    assert cat_features.is_cat is True
    assert cat_features.is_pad is False
    assert cat_features.num_channels == cat_features.batched_tensor.shape[-1]


def test_cat_features_to_pad(setup_cat_features):
    _, cat_features, _ = setup_cat_features

    # Test conversion to padded features
    pad_features = cat_features.to_pad()
    assert isinstance(pad_features, PadFeatures)
    assert pad_features.num_channels == cat_features.num_channels
    assert (pad_features.offsets == cat_features.offsets).all()

    # Test with specific padding multiple
    pad_multiple = 32
    pad_features_32 = cat_features.to_pad(pad_multiple=pad_multiple)
    assert pad_features_32.pad_multiple == pad_multiple
    assert pad_features_32.batched_tensor.shape[1] % pad_multiple == 0


def test_cat_features_equal_shape(setup_cat_features):
    _, cat_features, Ns = setup_cat_features

    # Test with same shape
    same_features = CatFeatures(torch.rand_like(cat_features.batched_tensor), cat_features.offsets)
    assert cat_features.equal_shape(same_features)

    # Test with different shape
    different_Ns = Ns + 1
    different_offsets = torch.tensor([0] + list(torch.cumsum(different_Ns, dim=0).numpy()))
    different_features = CatFeatures(
        torch.rand(different_offsets[-1], cat_features.num_channels), different_offsets
    )
    assert not cat_features.equal_shape(different_features)
