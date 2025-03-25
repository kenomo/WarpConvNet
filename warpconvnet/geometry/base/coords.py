# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor
from .batched import BatchedTensor


from warpconvnet.geometry.coords.search.search_results import RealSearchResult
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates


class Coords(BatchedTensor):
    """Base class for coordinates."""

    @property
    def num_spatial_dims(self):
        return self.batched_tensor.shape[1]  # tensor does not have batch index

    def neighbors(
        self,
        query_coords: "Coords",
        search_args: dict,
    ) -> "RealSearchResult":
        """
        Find the neighbors of the query_coords in the current coordinates.

        Args:
            query_coords: The coordinates to search for neighbors
            search_args: Arguments for the search
        """
        raise NotImplementedError

    @property
    def batch_indexed_coordinates(self) -> Tensor:
        return batch_indexed_coordinates(self.batched_tensor, self.offsets)
