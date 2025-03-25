# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.features.pad import PadFeatures
from warpconvnet.geometry.features.cat import CatFeatures


@pytest.fixture
def setup_pad_features():
    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    features = [torch.rand((N, C)) for N in Ns]
    pad_multiple = 32
    max_N = ((max(Ns) + pad_multiple - 1) // pad_multiple) * pad_multiple
    padded_features = torch.zeros(B, max_N, C)
    offsets = torch.tensor([0] + list(torch.cumsum(Ns, dim=0).numpy()))

    for i in range(B):
        padded_features[i, : Ns[i], :] = features[i]

    pad_features = PadFeatures(padded_features, offsets, pad_multiple)
    return features, pad_features, Ns, pad_multiple


def test_pad_features_init(setup_pad_features):
    features, pad_features, Ns, pad_multiple = setup_pad_features

    # Test initialization from list
    pad_from_list = PadFeatures(features, pad_multiple=pad_multiple)
    assert pad_from_list.num_channels == features[0].shape[1]
    assert (pad_from_list.offsets == pad_features.offsets).all()

    # Test initialization from 3D tensor
    pad_from_tensor = PadFeatures(pad_features.batched_tensor, pad_features.offsets, pad_multiple)
    assert (pad_from_tensor.batched_tensor == pad_features.batched_tensor).all()


def test_pad_features_properties(setup_pad_features):
    _, pad_features, _, _ = setup_pad_features

    assert pad_features.is_cat is False
    assert pad_features.is_pad is True
    assert pad_features.num_channels == pad_features.batched_tensor.shape[-1]
    assert pad_features.batch_size == pad_features.batched_tensor.shape[0]
    assert pad_features.max_num_points == pad_features.batched_tensor.shape[1]


def test_pad_features_to_cat(setup_pad_features):
    _, pad_features, _, _ = setup_pad_features

    # Test conversion to cat features
    cat_features = pad_features.to_cat()
    assert isinstance(cat_features, CatFeatures)
    assert cat_features.num_channels == pad_features.num_channels
    assert (cat_features.offsets == pad_features.offsets).all()

    # Convert back to pad and verify
    pad_features_round_trip = cat_features.to_pad(pad_features.pad_multiple)
    assert (pad_features_round_trip.batched_tensor == pad_features.batched_tensor).all()


def test_pad_features_clear_padding(setup_pad_features):
    _, pad_features, Ns, _ = setup_pad_features

    clear_value = -1.0
    cleared = pad_features.clear_padding(clear_value)

    # Check that valid data is unchanged
    for i in range(len(Ns)):
        valid_data = cleared.batched_tensor[i, : Ns[i]]
        assert not (valid_data == clear_value).any()

        # Check that padding is cleared
        padding = cleared.batched_tensor[i, Ns[i] :]
        assert (padding == clear_value).all()


def test_pad_features_equal_shape(setup_pad_features):
    _, pad_features, _, _ = setup_pad_features

    # Test with same shape
    same_features = PadFeatures(
        torch.rand_like(pad_features.batched_tensor),
        pad_features.offsets,
        pad_features.pad_multiple,
    )
    assert pad_features.equal_shape(same_features)

    # Test with different shape
    different_features = PadFeatures(
        torch.rand(
            pad_features.batch_size, pad_features.max_num_points + 32, pad_features.num_channels
        ),
        pad_features.offsets,
        pad_features.pad_multiple,
    )
    assert not pad_features.equal_shape(different_features)
