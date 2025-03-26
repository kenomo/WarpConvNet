# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
from jaxtyping import Float

from warpconvnet.geometry.base.features import Features
from warpconvnet.geometry.features.ops.convert import cat_to_pad


class CatFeatures(Features):
    def check(self):
        super().check()
        assert self.batched_tensor.ndim == 2, "Batched tensor must be 2D"
        assert (
            self.batched_tensor.shape[0] == self.offsets[-1]
        ), f"Offsets {self.offsets} does not match tensors {self.batched_tensor.shape}"

    @property
    def is_cat(self) -> bool:
        return True

    def to_pad(self, pad_multiple: Optional[int] = None) -> "PadFeatures":  # noqa: F821
        return cat_to_pad(self, pad_multiple=pad_multiple)

    def equal_shape(self, value: object) -> bool:
        if not isinstance(value, CatFeatures):
            return False
        return (
            (self.offsets == value.offsets).all()
            and self.numel() == value.numel()
            and self.num_channels == value.num_channels
        )
