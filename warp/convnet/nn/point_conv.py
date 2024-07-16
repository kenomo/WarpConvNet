from typing import List, Literal, Optional

import torch
import torch.nn as nn

from warp.convnet.geometry.ops.neighbor_search_continuous import (
    NEIGHBOR_SEARCH_MODE,
    NeighborSearchArgs,
)
from warp.convnet.geometry.ops.point_pool import FeaturePoolingArgs
from warp.convnet.geometry.point_collection import (
    BatchedCoordinates,
    BatchedFeatures,
    PointCollection,
)
from warp.convnet.nn.base_module import BaseModel, BaseModule
from warp.convnet.nn.encoding import SinusoidalEncoding
from warp.convnet.nn.mlp import MLPBlock
from warp.convnet.nn.point_transform import PointCollectionMLP, PointCollectionTransform
from warp.convnet.ops.reductions import REDUCTION_TYPES, row_reduction

__all__ = ["PointConv", "PointConvBlock", "PointConvUNetBlock", "PointConvUNet"]


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
        out_str = f"{self.__class__.__name__}(in_channels={self.in_channels} out_channels={self.out_channels}"
        if self.downsample_voxel_size is not None:
            out_str += f" down_voxel_size={self.downsample_voxel_size}"
        if self.use_rel_pos_encode:
            out_str += f" rel_pos_encode={self.use_rel_pos_encode}"
        out_str += ")"
        return out_str

    def forward(
        self,
        in_point_features: PointCollection,
        query_point_features: Optional[PointCollection] = None,
    ) -> PointCollection:
        """
        When out_point_features is None, the output will be generated on the
        in_point_features.batched_coordinates.
        """
        if self.out_point_feature_type == "provided":
            assert (
                query_point_features is not None
            ), "query_point_features must be provided for the provided type"
        elif self.out_point_feature_type == "downsample":
            assert query_point_features is None
            query_point_features = in_point_features.voxel_downsample(
                self.downsample_voxel_size,
                pooling_args=self.pooling_args,
            )
        elif self.out_point_feature_type == "same":
            assert query_point_features is None
            query_point_features = in_point_features

        in_num_channels = in_point_features.num_channels
        query_num_channels = query_point_features.num_channels
        assert (
            in_num_channels
            + query_num_channels
            + self.use_rel_pos_encode * self.positional_encoding.num_channels * 3
            + (not self.use_rel_pos_encode) * self.use_rel_pos * 3
            == self.edge_transform_mlp.in_channels
        ), f"input features shape {in_point_features.feature_tensor.shape} and query feature shape {query_point_features.feature_tensor.shape} does not match the edge_transform_mlp input features {self.edge_transform_mlp.in_channels}"

        # Get the neighbors
        neighbors = in_point_features.neighbors(
            query_coords=query_point_features.batched_coordinates,
            search_args=self.neighbor_search_args,
        )
        neighbors_index = neighbors.neighbors_index.long().view(-1)
        neighbors_row_splits = neighbors.neighbors_row_splits
        num_reps = neighbors_row_splits[1:] - neighbors_row_splits[:-1]

        # repeat the self features using num_reps
        rep_in_features = in_point_features.feature_tensor[neighbors_index]
        self_features = torch.repeat_interleave(
            query_point_features.feature_tensor.view(-1, query_num_channels).contiguous(),
            num_reps,
            dim=0,
        )
        edge_features = [rep_in_features, self_features]
        if self.use_rel_pos or self.use_rel_pos_encode:
            in_rep_vertices = in_point_features.coordinate_tensor.view(-1, 3)[neighbors_index]
            self_vertices = torch.repeat_interleave(
                query_point_features.coordinate_tensor.view(-1, 3).contiguous(), num_reps, dim=0
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
                batched_tensor=query_point_features.coordinate_tensor,
                offsets=query_point_features.offsets,
            ),
            batched_features=BatchedFeatures(
                batched_tensor=out_features,
                offsets=query_point_features.offsets,
            ),
        )


class PointConvBlock(BaseModule):
    """
    A conv block that consists of two consecutive PointConv, activation, and normalization and a skip connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neighbor_search_args: NeighborSearchArgs,
        pooling_args: Optional[FeaturePoolingArgs] = None,
        edge_transform_mlp: Optional[nn.Module] = None,
        out_transform_mlp: Optional[nn.Module] = None,
        intermediate_dim: Optional[int] = None,
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
        norm_layer1: Optional[nn.Module] = None,
        norm_layer2: Optional[nn.Module] = None,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = out_channels
        self.point_conv1 = PointConv(
            in_channels=in_channels,
            out_channels=intermediate_dim,
            neighbor_search_args=neighbor_search_args,
            pooling_args=pooling_args,
            edge_transform_mlp=edge_transform_mlp,
            out_transform_mlp=out_transform_mlp,
            hidden_dim=hidden_dim,
            channel_multiplier=channel_multiplier,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            pos_encode_range=pos_encode_range,
            reductions=reductions,
            downsample_voxel_size=downsample_voxel_size,
            out_point_feature_type=out_point_feature_type,
            provided_in_channels=provided_in_channels,
        )
        self.point_conv2 = PointConv(
            in_channels=intermediate_dim,
            out_channels=out_channels,
            neighbor_search_args=neighbor_search_args,
            pooling_args=pooling_args,
            edge_transform_mlp=edge_transform_mlp,
            out_transform_mlp=out_transform_mlp,
            hidden_dim=hidden_dim,
            channel_multiplier=channel_multiplier,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            pos_encode_range=pos_encode_range,
            reductions=reductions,
            downsample_voxel_size=downsample_voxel_size,
            out_point_feature_type=out_point_feature_type,
            provided_in_channels=provided_in_channels,
        )

        if norm_layer1 is None:
            norm_layer1 = nn.LayerNorm(intermediate_dim)
        if norm_layer2 is None:
            norm_layer2 = nn.LayerNorm(out_channels)
        if activation is None:
            activation = nn.ReLU()

        self.norm1 = PointCollectionTransform(norm_layer1)
        self.norm2 = PointCollectionTransform(norm_layer2)
        self.relu = PointCollectionTransform(activation)

    def forward(self, in_point_features: PointCollection) -> PointCollection:
        out1 = self.point_conv1(in_point_features)
        out1 = self.norm1(out1)
        out1 = self.relu(out1)
        out2 = self.point_conv2(out1)
        out2 = self.norm2(out2)
        # Skip connection
        out2 = out2 + out1
        out2 = self.relu(out2)
        return PointCollection(
            batched_coordinates=out2.batched_coordinates,
            batched_features=out2.batched_features,
        )


class PointConvUNetBlock(BaseModule):
    """
    Given an input module, the UNet block will return a list of point collections at each level of the UNet from the inner module.
    """

    def __init__(
        self,
        in_channels: int,
        inner_module_in_channels: int,
        inner_module_out_channels: int,
        out_channels: int,
        neighbor_search_args: NeighborSearchArgs,
        pooling_args: Optional[FeaturePoolingArgs] = None,
        inner_module: BaseModule = None,
        num_down_blocks: int = 1,
        num_up_blocks: int = 0,
        edge_transform_mlp: Optional[nn.Module] = None,
        out_transform_mlp: Optional[nn.Module] = None,
        intermediate_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        channel_multiplier: int = 2,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        pos_encode_range: float = 4,
        reductions: List[REDUCTION_TYPES] = ("mean",),
        downsample_voxel_size: Optional[float] = None,
    ):
        assert inner_module is None or isinstance(inner_module, PointConvUNetBlock)
        assert downsample_voxel_size > 0

        super().__init__()
        down_conv_block = [
            PointConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                neighbor_search_args=neighbor_search_args,
                pooling_args=pooling_args,
                edge_transform_mlp=edge_transform_mlp,
                out_transform_mlp=out_transform_mlp,
                intermediate_dim=intermediate_dim,
                hidden_dim=hidden_dim,
                channel_multiplier=channel_multiplier,
                use_rel_pos=use_rel_pos,
                use_rel_pos_encode=use_rel_pos_encode,
                pos_encode_dim=pos_encode_dim,
                pos_encode_range=pos_encode_range,
                reductions=reductions,
                out_point_feature_type="same",
            )
        ]

        for _ in range(num_down_blocks):
            down_conv_block.append(
                PointConvBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    neighbor_search_args=neighbor_search_args,
                    pooling_args=pooling_args,
                    edge_transform_mlp=edge_transform_mlp,
                    out_transform_mlp=out_transform_mlp,
                    intermediate_dim=intermediate_dim,
                    hidden_dim=hidden_dim,
                    channel_multiplier=channel_multiplier,
                    use_rel_pos=use_rel_pos,
                    use_rel_pos_encode=use_rel_pos_encode,
                    pos_encode_dim=pos_encode_dim,
                    pos_encode_range=pos_encode_range,
                    reductions=reductions,
                    out_point_feature_type="same",
                )
            )

        self.down_conv_block = nn.Sequential(*down_conv_block)

        self.down_conv = PointConv(
            in_channels=in_channels,
            out_channels=inner_module_in_channels,
            neighbor_search_args=neighbor_search_args,
            pooling_args=pooling_args,
            edge_transform_mlp=edge_transform_mlp,
            out_transform_mlp=out_transform_mlp,
            hidden_dim=hidden_dim,
            channel_multiplier=channel_multiplier,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            pos_encode_range=pos_encode_range,
            reductions=reductions,
            out_point_feature_type="downsample",
            downsample_voxel_size=downsample_voxel_size,
        )

        self.inner_module = inner_module

        self.up_conv = PointConv(
            in_channels=inner_module_out_channels,
            out_channels=out_channels,
            neighbor_search_args=neighbor_search_args,
            pooling_args=pooling_args,
            edge_transform_mlp=edge_transform_mlp,
            out_transform_mlp=out_transform_mlp,
            hidden_dim=hidden_dim,
            channel_multiplier=channel_multiplier,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            pos_encode_range=pos_encode_range,
            reductions=reductions,
            out_point_feature_type="provided",
            provided_in_channels=in_channels,
        )

        if num_up_blocks > 0:
            self.up_conv_block = nn.Sequential(
                *[
                    PointConvBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        neighbor_search_args=neighbor_search_args,
                        pooling_args=pooling_args,
                        edge_transform_mlp=edge_transform_mlp,
                        out_transform_mlp=out_transform_mlp,
                        intermediate_dim=intermediate_dim,
                        hidden_dim=hidden_dim,
                        channel_multiplier=channel_multiplier,
                        use_rel_pos=use_rel_pos,
                        use_rel_pos_encode=use_rel_pos_encode,
                        pos_encode_dim=pos_encode_dim,
                        pos_encode_range=pos_encode_range,
                        reductions=reductions,
                        out_point_feature_type="same",
                    )
                    for _ in range(num_up_blocks)
                ]
            )
        else:
            self.up_conv_block = nn.Identity()

        if in_channels != out_channels:
            self.skip = PointCollectionMLP(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=max(in_channels, out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, point_collection: PointCollection) -> List[PointCollection]:
        """
        Given an input point collection, the network will return a list of point collections at each level of the UNet.
        """
        out_down = self.down_conv_block(point_collection)
        out_down = self.down_conv(out_down)
        if self.inner_module is None:
            out_inner: List[PointCollection] = [out_down]
        else:
            out_inner: List[PointCollection] = self.inner_module(out_down)
        out_up = self.up_conv(out_inner[0], point_collection)
        out_up = out_up + self.skip(point_collection)
        out_up = self.up_conv_block(out_up)
        return [out_up] + out_inner


class PointConvUNet(BaseModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_channels: List[int],
        up_channels: List[int],
        downsample_voxel_size: float | List[float],
        neighbor_search_args: NeighborSearchArgs,
        voxel_size_multiplier: float = 2.0,
        pooling_args: Optional[FeaturePoolingArgs] = None,
        edge_transform_mlp: Optional[nn.Module] = None,
        out_transform_mlp: Optional[nn.Module] = None,
        intermediate_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        channel_multiplier: int = 2,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        pos_encode_range: float = 4,
        reductions: List[REDUCTION_TYPES] = ("mean",),
        num_levels: int = 4,
    ):
        super().__init__()
        assert len(down_channels) > num_levels and len(up_channels) > num_levels
        self.num_levels = num_levels

        if isinstance(downsample_voxel_size, float):
            downsample_voxel_size = [
                downsample_voxel_size * voxel_size_multiplier**i for i in range(num_levels)
            ]

        print(downsample_voxel_size)

        self.in_map = PointCollectionTransform(nn.Linear(in_channels, down_channels[0]))

        # Create from the deepest level to the shallowest level
        inner_block = None
        for i in range(num_levels - 1, -1, -1):
            curr_neighbor_search_args: NeighborSearchArgs = neighbor_search_args.clone(
                radius=downsample_voxel_size[i]
            )
            inner_block = PointConvUNetBlock(
                inner_module=inner_block,
                in_channels=down_channels[i],
                inner_module_in_channels=down_channels[i + 1],
                inner_module_out_channels=up_channels[i + 1],
                out_channels=up_channels[i],
                neighbor_search_args=curr_neighbor_search_args,
                pooling_args=pooling_args,
                edge_transform_mlp=edge_transform_mlp,
                out_transform_mlp=out_transform_mlp,
                intermediate_dim=intermediate_dim,
                hidden_dim=hidden_dim,
                channel_multiplier=channel_multiplier,
                use_rel_pos=use_rel_pos,
                use_rel_pos_encode=use_rel_pos_encode,
                pos_encode_dim=pos_encode_dim,
                pos_encode_range=pos_encode_range,
                reductions=reductions,
                downsample_voxel_size=downsample_voxel_size[i],
            )
        self.unet = inner_block

        self.out_map = PointCollectionTransform(nn.Linear(up_channels[0], out_channels))

    def forward(self, point_collection: PointCollection) -> List[PointCollection]:
        """
        Given an input point collection, the network will return a list of point collections at each level of the UNet.
        """
        out = self.in_map(point_collection)
        out = self.unet(out)
        out[0] = self.out_map(out[0])
        return out
