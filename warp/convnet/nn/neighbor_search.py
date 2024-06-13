from typing import Optional
from jaxtyping import Float, Int

import torch
from torch import Tensor


class NeighborRadiusSearchLayer(torch.nn.Module):
    """NeighborRadiusSearchLayer."""

    def __init__(
        self,
        radius: Optional[float] = None,
    ):
        super().__init__()
        self.radius = radius

    @torch.no_grad()
    def forward(
        self,
        ref_positions: Int[Tensor, "N 3"],
        query_positions: Int[Tensor, "M 3"],
        radius: Optional[float] = None,
    ) -> NeighborSearchReturn:
        if radius is None:
            radius = self.radius
        return neighbor_radius_search(ref_positions, query_positions, radius)


class NeighborPoolingLayer(torch.nn.Module):
    """NeighborPoolingLayer."""

    def __init__(self, reduction: REDUCTION_TYPES = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, in_features: Float[Tensor, "N C"], neighbors: NeighborSearchReturn
    ) -> Float[Tensor, "M C"]:
        """
        inp_features: [N,C]
        neighbors: NeighborSearchReturn. If None, will be computed. For the same inp_positions and out_positions, this can be reused.
        """
        rep_features = in_features[neighbors.neighbors_index.long()]
        out_features = row_reduction(
            rep_features, neighbors.neighbors_row_splits, reduction=self.reduction
        )
        return out_features


class NeighborRadiusSearchLayer(torch.nn.Module):
    """NeighborRadiusSearchLayer."""

    def __init__(
        self,
        radius: Optional[float] = None,
    ):
        super().__init__()
        self.radius = radius

    @torch.no_grad()
    def forward(
        self,
        ref_positions: Int[Tensor, "N 3"],
        query_positions: Int[Tensor, "M 3"],
        radius: Optional[float] = None,
    ) -> NeighborSearchReturn:
        if radius is None:
            radius = self.radius
        return neighbor_radius_search(ref_positions, query_positions, radius)


class NeighborPoolingLayer(torch.nn.Module):
    """NeighborPoolingLayer."""

    def __init__(self, reduction: REDUCTION_TYPES = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, in_features: Float[Tensor, "N C"], neighbors: NeighborSearchReturn
    ) -> Float[Tensor, "M C"]:
        """
        inp_features: [N,C]
        neighbors: NeighborSearchReturn. If None, will be computed. For the same inp_positions and out_positions, this can be reused.
        """
        rep_features = in_features[neighbors.neighbors_index.long()]
        out_features = row_reduction(
            rep_features, neighbors.neighbors_row_splits, reduction=self.reduction
        )
        return out_features
