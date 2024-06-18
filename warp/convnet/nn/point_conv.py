from typing import List, Literal, Optional

import torch
import torch.nn as nn

from warp.convnet.geometry.ops.neighbor_search import (
    NEIGHBOR_SEARCH_MODE,
    NeighborSearchArgs,
)
from warp.convnet.geometry.ops.point_pool import FeaturePoolingArgs
from warp.convnet.geometry.point_collection import (
    BatchedCoordinates,
    BatchedFeatures,
    PointCollection,
)
from warp.convnet.nn.base_module import BaseModule
from warp.convnet.nn.encoding import SinusoidalEncoding
from warp.convnet.nn.mlp import MLPBlock
from warp.convnet.ops.reductions import REDUCTION_TYPES, row_reduction


class PointConv(BaseModule):
    """PointFeatureConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neighbor_search_args: NeighborSearchArgs,
        pooling_args: Optional[FeaturePoolingArgs] = None,
        edge_transform_mlp: Optional[nn.Module] = None,
        out_transform_mlp: Optional[nn.Module] = None,
        hidden_dim: Optional[int] = None,
        channel_multiplier: int = 2,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        pos_encode_range: float = 4,
        reductions: List[REDUCTION_TYPES] = ("mean",),
        downsample_voxel_size: Optional[float] = None,
        out_point_feature_type: Literal["provided", "downsample", "same"] = "same",
        provided_in_channels: Optional[int] = None,
    ):
        """If use_relative_position_encoding is True, the positional encoding vertex coordinate
        difference is added to the edge features.

        downsample_voxel_size: If not None, the input point cloud will be downsampled.

        out_point_feature_type: If "upsample", the output point features will be upsampled to the input point cloud size.

        use_rel_pos: If True, the relative position of the neighbor points will be used as the edge features.
        use_rel_pos_encode: If True, the encoding relative position of the neighbor points will be used as the edge features.
        """
        super().__init__()
        assert (
            isinstance(reductions, (tuple, list)) and len(reductions) > 0
        ), f"reductions must be a list or tuple of length > 0, got {reductions}"
        if out_point_feature_type == "provided":
            assert (
                downsample_voxel_size is None
            ), "downsample_voxel_size is only used for downsample"
            assert (
                provided_in_channels is not None
            ), "provided_in_channels must be provided for provided type"
        elif out_point_feature_type == "downsample":
            assert (
                downsample_voxel_size is not None
            ), "downsample_voxel_size must be provided for downsample"
            assert (
                provided_in_channels is None
            ), "provided_in_channels must be None for downsample type"
            assert pooling_args is not None, "pooling_args must be provided for downsample type"
        elif out_point_feature_type == "same":
            assert (
                downsample_voxel_size is None
            ), "downsample_voxel_size is only used for downsample"
            assert provided_in_channels is None, "provided_in_channels must be None for same type"
        if (
            downsample_voxel_size is not None
            and neighbor_search_args.mode == NEIGHBOR_SEARCH_MODE.RADIUS
            and downsample_voxel_size > neighbor_search_args.radius
        ):
            raise ValueError(
                f"downsample_voxel_size {downsample_voxel_size} must be <= radius {neighbor_search_args.radius}"
            )

        assert isinstance(neighbor_search_args, NeighborSearchArgs)
        self.reductions = reductions
        self.downsample_voxel_size = downsample_voxel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_rel_pos = use_rel_pos
        self.use_rel_pos_encode = use_rel_pos_encode
        self.out_point_feature_type = out_point_feature_type
        self.neighbor_search_args = neighbor_search_args
        self.pooling_args = pooling_args
        self.positional_encoding = SinusoidalEncoding(pos_encode_dim, data_range=pos_encode_range)
        # When down voxel size is not None, there will be out_point_features will be provided as an additional input
        if provided_in_channels is None:
            provided_in_channels = in_channels
        if hidden_dim is None:
            hidden_dim = channel_multiplier * out_channels
        if edge_transform_mlp is None:
            edge_in_channels = in_channels + provided_in_channels
            if use_rel_pos_encode:
                edge_in_channels += pos_encode_dim * 3
            elif use_rel_pos:
                edge_in_channels += 3
            edge_transform_mlp = MLPBlock(
                in_channels=edge_in_channels,
                hidden_channels=hidden_dim,
                out_channels=out_channels,
            )
        self.edge_transform_mlp = edge_transform_mlp
        if out_transform_mlp is None:
            out_transform_mlp = MLPBlock(
                in_channels=out_channels * len(reductions),
                hidden_channels=hidden_dim,
                out_channels=out_channels,
            )
        self.out_transform_mlp = out_transform_mlp

    def __repr__(self):
        out_str = f"{self.__class__.__name__}(in_channels={self.in_channels} out_channels={self.out_channels} neighbors={self.neighbor_search_args} reductions={self.reductions}"
        if self.downsample_voxel_size is not None:
            out_str += f" down_voxel_size={self.downsample_voxel_size}"
        if self.use_rel_pos_encode:
            out_str += f" rel_pos_encode={self.use_rel_pos_encode}"
        out_str += ")"
        return out_str

    def forward(
        self,
        in_point_features: PointCollection,
        out_point_features: Optional[PointCollection] = None,
    ) -> PointCollection:
        """
        When out_point_features is None, the output will be generated on the
        in_point_features.batched_coordinates.
        """
        if self.out_point_feature_type == "provided":
            assert (
                out_point_features is not None
            ), "out_point_features must be provided for the provided type"
        elif self.out_point_feature_type == "downsample":
            assert out_point_features is None
            out_point_features = in_point_features.voxel_downsample(
                self.downsample_voxel_size,
                pooling_args=self.pooling_args,
            )
        elif self.out_point_feature_type == "same":
            assert out_point_features is None
            out_point_features = in_point_features

        in_num_channels = in_point_features.num_channels
        out_num_channels = out_point_features.num_channels
        assert (
            in_num_channels
            + out_num_channels
            + self.use_rel_pos_encode * self.positional_encoding.num_channels * 3
            + (not self.use_rel_pos_encode) * self.use_rel_pos * 3
            == self.edge_transform_mlp.in_channels
        ), f"input features shape {in_point_features.features.shape} and {out_point_features.features.shape} does not match the edge_transform_mlp input features {self.edge_transform_mlp.in_channels}"

        # Get the neighbors
        neighbors = in_point_features.batched_coordinates.neighbors(
            query_coords=out_point_features.batched_coordinates,
            search_args=self.neighbor_search_args,
        )
        neighbors_index = neighbors.neighbors_index.long().view(-1)
        neighbors_row_splits = neighbors.neighbors_row_splits
        num_reps = neighbors_row_splits[1:] - neighbors_row_splits[:-1]

        # repeat the self features using num_reps
        rep_in_features = in_point_features.features[neighbors_index]
        self_features = torch.repeat_interleave(
            out_point_features.features.view(-1, out_num_channels).contiguous(),
            num_reps,
            dim=0,
        )
        edge_features = [rep_in_features, self_features]
        if self.use_rel_pos or self.use_rel_pos_encode:
            in_rep_vertices = in_point_features.coords.view(-1, 3)[neighbors_index]
            self_vertices = torch.repeat_interleave(
                out_point_features.coords.view(-1, 3).contiguous(), num_reps, dim=0
            )
            if self.use_rel_pos_encode:
                edge_features.append(
                    self.positional_encoding(
                        in_rep_vertices.view(-1, 3) - self_vertices.view(-1, 3)
                    )
                )
            elif self.use_rel_pos:
                edge_features.append(in_rep_vertices - self_vertices)
        edge_features = torch.cat(edge_features, dim=1)
        edge_features = self.edge_transform_mlp(edge_features)
        # if in_weight is not None:
        #     assert in_weight.shape[0] == in_point_features.features.shape[0]
        #     rep_weights = in_weight[neighbors_index]
        #     edge_features = edge_features * rep_weights.squeeze().unsqueeze(-1)

        out_features = []
        for reduction in self.reductions:
            out_features.append(
                row_reduction(edge_features, neighbors_row_splits, reduction=reduction)
            )
        out_features = torch.cat(out_features, dim=-1)
        out_features = self.out_transform_mlp(out_features)

        return PointCollection(
            batched_coordinates=BatchedCoordinates(
                batched_tensor=out_point_features.coords,
                offsets=out_point_features.offsets,
            ),
            batched_features=BatchedFeatures(
                batched_tensor=out_features,
                offsets=out_point_features.offsets,
            ),
        )
