import warnings
from typing import List, Literal, Optional

import torch
import torch.nn as nn

from warpconvnet.geometry.ops.neighbor_search_continuous import (
    CONTINUOUS_NEIGHBOR_SEARCH_MODE,
    ContinuousNeighborSearchArgs,
)
from warpconvnet.geometry.point_collection import (
    BatchedCoordinates,
    BatchedFeatures,
    PointCollection,
)
from warpconvnet.nn.base_module import BaseModule
from warpconvnet.nn.encoding import SinusoidalEncoding
from warpconvnet.nn.functional.point_pool import FeaturePoolingArgs
from warpconvnet.nn.mlp import MLPBlock
from warpconvnet.nn.point_transform import (
    PointCollectionLinear,
    PointCollectionTransform,
)
from warpconvnet.ops.reductions import REDUCTION_TYPES_STR, row_reduction

__all__ = ["PointConv", "PointConvBlock", "PointConvUNetBlock"]


class PointConv(BaseModule):
    """PointFeatureConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neighbor_search_args: ContinuousNeighborSearchArgs,
        pooling_args: Optional[FeaturePoolingArgs] = None,
        edge_transform_mlp: Optional[nn.Module] = None,
        out_transform_mlp: Optional[nn.Module] = None,
        hidden_dim: Optional[int] = None,
        channel_multiplier: int = 2,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        pos_encode_range: float = 4,
        reductions: List[REDUCTION_TYPES_STR] = ("mean",),
        out_point_type: Literal["provided", "downsample", "same"] = "same",
        provided_in_channels: Optional[int] = None,
    ):
        """If use_relative_position_encoding is True, the positional encoding vertex coordinate
        difference is added to the edge features.

        out_point_feature_type: If "upsample", the output point features will be upsampled to the input point cloud size.

        use_rel_pos: If True, the relative position of the neighbor points will be used as the edge features.
        use_rel_pos_encode: If True, the encoding relative position of the neighbor points will be used as the edge features.
        """
        super().__init__()
        assert (
            isinstance(reductions, (tuple, list)) and len(reductions) > 0
        ), f"reductions must be a list or tuple of length > 0, got {reductions}"
        if out_point_type == "provided":
            assert pooling_args is None
            assert (
                provided_in_channels is not None
            ), "provided_in_channels must be provided for provided type"
        elif out_point_type == "downsample":
            assert pooling_args is not None, "pooling_args must be provided for downsample type"
            assert (
                provided_in_channels is None
            ), "provided_in_channels must be None for downsample type"
            # print warning if search radius is not \sqrt(3) times the downsample voxel size
            if (
                pooling_args.downsample_voxel_size is not None
                and neighbor_search_args.mode == CONTINUOUS_NEIGHBOR_SEARCH_MODE.RADIUS
                and neighbor_search_args.radius < pooling_args.downsample_voxel_size * (3**0.5)
            ):
                warnings.warn(
                    f"neighbor search radius {neighbor_search_args.radius} is less than sqrt(3) times the downsample voxel size {pooling_args.downsample_voxel_size}",
                    stacklevel=2,
                )
        elif out_point_type == "same":
            assert pooling_args is None, "pooling_args is only used for downsample"
            assert provided_in_channels is None, "provided_in_channels must be None for same type"
        if (
            pooling_args is not None
            and pooling_args.downsample_voxel_size is not None
            and neighbor_search_args.mode == CONTINUOUS_NEIGHBOR_SEARCH_MODE.RADIUS
            and pooling_args.downsample_voxel_size > neighbor_search_args.radius
        ):
            raise ValueError(
                f"downsample_voxel_size {pooling_args.downsample_voxel_size} must be <= radius {neighbor_search_args.radius}"
            )

        assert isinstance(neighbor_search_args, ContinuousNeighborSearchArgs)
        self.reductions = reductions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_rel_pos = use_rel_pos
        self.use_rel_pos_encode = use_rel_pos_encode
        self.out_point_feature_type = out_point_type
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
        if self.use_rel_pos_encode:
            out_str += f" rel_pos_encode={self.use_rel_pos_encode}"
        if self.pooling_args is not None:
            out_str += f" pooling={self.pooling_args}"
        if self.neighbor_search_args is not None:
            out_str += f" neighbor={self.neighbor_search_args}"
        out_str += ")"
        return out_str

    def forward(
        self,
        in_pc: PointCollection,
        query_pc: Optional[PointCollection] = None,
    ) -> PointCollection:
        """
        When out_point_features is None, the output will be generated on the
        in_point_features.batched_coordinates.
        """
        if self.out_point_feature_type == "provided":
            assert (
                query_pc is not None
            ), "query_point_features must be provided for the provided type"
        elif self.out_point_feature_type == "downsample":
            assert query_pc is None
            query_pc = in_pc.voxel_downsample(
                self.pooling_args.downsample_voxel_size,
                pooling_args=self.pooling_args,
            )
        elif self.out_point_feature_type == "same":
            assert query_pc is None
            query_pc = in_pc

        in_num_channels = in_pc.num_channels
        query_num_channels = query_pc.num_channels
        assert (
            in_num_channels
            + query_num_channels
            + self.use_rel_pos_encode * self.positional_encoding.num_channels * 3
            + (not self.use_rel_pos_encode) * self.use_rel_pos * 3
            == self.edge_transform_mlp.in_channels
        ), f"input features shape {in_pc.feature_tensor.shape} and query feature shape {query_pc.feature_tensor.shape} does not match the edge_transform_mlp input features {self.edge_transform_mlp.in_channels}"

        # Get the neighbors
        neighbors = in_pc.neighbors(
            query_coords=query_pc.batched_coordinates,
            search_args=self.neighbor_search_args,
        )
        neighbors_index = neighbors.neighbors_index.long().view(-1)
        neighbors_row_splits = neighbors.neighbors_row_splits
        num_reps = neighbors_row_splits[1:] - neighbors_row_splits[:-1]

        # repeat the self features using num_reps
        rep_in_features = in_pc.feature_tensor[neighbors_index]
        self_features = torch.repeat_interleave(
            query_pc.feature_tensor.view(-1, query_num_channels).contiguous(),
            num_reps,
            dim=0,
        )
        edge_features = [rep_in_features, self_features]
        if self.use_rel_pos or self.use_rel_pos_encode:
            in_rep_vertices = in_pc.coordinate_tensor.view(-1, 3)[neighbors_index]
            self_vertices = torch.repeat_interleave(
                query_pc.coordinate_tensor.view(-1, 3).contiguous(),
                num_reps,
                dim=0,
            )
            rel_coords = in_rep_vertices.view(-1, 3) - self_vertices.view(-1, 3)
            if self.use_rel_pos_encode:
                edge_features.append(self.positional_encoding(rel_coords))
            elif self.use_rel_pos:
                edge_features.append(rel_coords)
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
                batched_tensor=query_pc.coordinate_tensor,
                offsets=query_pc.offsets,
            ),
            batched_features=BatchedFeatures(
                batched_tensor=out_features,
                offsets=query_pc.offsets,
            ),
            **query_pc.extra_attributes,
        )


class PointConvBlock(BaseModule):
    """
    A conv block that consists of two consecutive PointConv, activation, and normalization and a skip connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neighbor_search_args: ContinuousNeighborSearchArgs,
        edge_transform_mlp: Optional[nn.Module] = None,
        out_transform_mlp: Optional[nn.Module] = None,
        intermediate_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        channel_multiplier: int = 2,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        pos_encode_range: float = 4,
        reductions: List[REDUCTION_TYPES_STR] = ("mean",),
        out_point_feature_type: Literal["provided", "downsample", "same"] = "same",
        provided_in_channels: Optional[int] = None,
        norm_layer1: Optional[nn.Module] = None,
        norm_layer2: Optional[nn.Module] = None,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert out_point_feature_type == "same", "Only same type is supported for now"
        if intermediate_dim is None:
            intermediate_dim = out_channels
        self.point_conv1 = PointConv(
            in_channels=in_channels,
            out_channels=intermediate_dim,
            neighbor_search_args=neighbor_search_args,
            edge_transform_mlp=edge_transform_mlp,
            out_transform_mlp=out_transform_mlp,
            hidden_dim=hidden_dim,
            channel_multiplier=channel_multiplier,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            pos_encode_range=pos_encode_range,
            reductions=reductions,
            out_point_type="same",
            provided_in_channels=provided_in_channels,
        )
        self.point_conv2 = PointConv(
            in_channels=intermediate_dim,
            out_channels=out_channels,
            neighbor_search_args=neighbor_search_args,
            edge_transform_mlp=edge_transform_mlp,
            out_transform_mlp=out_transform_mlp,
            hidden_dim=hidden_dim,
            channel_multiplier=channel_multiplier,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            pos_encode_range=pos_encode_range,
            reductions=reductions,
            out_point_type="same",
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

    def forward(self, in_pc: PointCollection) -> PointCollection:
        out1 = self.point_conv1(in_pc)
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
            **out2.extra_attributes,
        )


class PointConvEncoder(BaseModule):
    """
    Generate a list of output point collections from each level of the encoder
    given an input point collection.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        num_blocks_per_level: List[int] | int,
        neighbor_search_args: ContinuousNeighborSearchArgs,
        neighbor_search_radii: List[float],
        pooling_args: Optional[FeaturePoolingArgs],
        downsample_voxel_sizes: List[float],
        channel_multiplier: int = 2,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        pos_encode_range: float = 4,
        reductions: List[REDUCTION_TYPES_STR] = ("mean",),
        num_levels: int = 4,
    ):
        super().__init__()
        self.num_levels = num_levels
        if isinstance(num_blocks_per_level, int):
            num_blocks_per_level = [num_blocks_per_level] * num_levels
        assert len(encoder_channels) == num_levels + 1
        assert len(num_blocks_per_level) == self.num_levels
        assert len(downsample_voxel_sizes) == self.num_levels
        assert len(neighbor_search_radii) == self.num_levels
        for level in range(self.num_levels - 1):
            assert downsample_voxel_sizes[level] < downsample_voxel_sizes[level + 1]
            assert neighbor_search_radii[level] < neighbor_search_radii[level + 1]
            # print warning if search radius is not \sqrt(3) times the downsample voxel size
            if neighbor_search_args.mode == CONTINUOUS_NEIGHBOR_SEARCH_MODE.RADIUS:
                assert neighbor_search_radii[level] > downsample_voxel_sizes[level] * (
                    3**0.5
                ), f"neighbor search radius {neighbor_search_radii[level]} is less than sqrt(3) times the downsample voxel size {downsample_voxel_sizes[level]} at level {level}"

        self.down_blocks = nn.ModuleList()

        for level in range(self.num_levels):
            in_channels = encoder_channels[level]
            out_channels = encoder_channels[level + 1]
            down_neighbor_search_args: ContinuousNeighborSearchArgs = neighbor_search_args.clone(
                radius=2 * downsample_voxel_sizes[level]
            )
            down_pooling_args: FeaturePoolingArgs = pooling_args.clone(
                downsample_voxel_size=downsample_voxel_sizes[level]
            )
            # Fisrt block out_point_feature_type is downsample, rest are conv blocks are "same"
            down_block = [
                PointConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    neighbor_search_args=down_neighbor_search_args,
                    pooling_args=down_pooling_args,
                    use_rel_pos=use_rel_pos,
                    use_rel_pos_encode=use_rel_pos_encode,
                    pos_encode_dim=pos_encode_dim,
                    pos_encode_range=pos_encode_range,
                    reductions=reductions,
                    out_point_type="downsample",
                )
            ]

            curr_neighbor_search_args: ContinuousNeighborSearchArgs = neighbor_search_args.clone(
                radius=neighbor_search_radii[level]
            )
            for _ in range(num_blocks_per_level[level]):
                down_block.append(
                    PointConvBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        neighbor_search_args=curr_neighbor_search_args,
                        channel_multiplier=channel_multiplier,
                        use_rel_pos=use_rel_pos,
                        use_rel_pos_encode=use_rel_pos_encode,
                        pos_encode_dim=pos_encode_dim,
                        pos_encode_range=pos_encode_range,
                        reductions=reductions,
                        out_point_feature_type="same",
                    )
                )

            self.down_blocks.append(nn.Sequential(*down_block))

    def forward(self, in_point_features: PointCollection) -> List[PointCollection]:
        out_point_features = []
        for down_block in self.down_blocks:
            out_point_features.append(in_point_features)
            in_point_features = down_block(in_point_features)
        out_point_features.append(in_point_features)
        return out_point_features


class PointConvDecoder(BaseModule):
    def __init__(
        self,
        decoder_channels: List[int],  # descending
        encoder_channels: List[int],  # ascending
        num_blocks_per_level: List[int],
        neighbor_search_args: ContinuousNeighborSearchArgs,
        neighbor_search_radii: List[float],  # descending
        channel_multiplier: int = 2,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        pos_encode_range: float = 4,
        reductions: List[REDUCTION_TYPES_STR] = ("mean",),
        num_levels: int = 4,
    ):
        super().__init__()
        self.num_levels = num_levels
        assert len(decoder_channels) == num_levels + 1
        assert len(num_blocks_per_level) == self.num_levels
        # Comming from encoder, the encoder_channels are in ascending order
        assert len(encoder_channels) >= num_levels + 1
        assert len(neighbor_search_radii) >= self.num_levels
        for level in range(self.num_levels - 1):
            assert (
                neighbor_search_radii[level] > neighbor_search_radii[level + 1]
            ), f"neighbor search radius must be in descending order, got {neighbor_search_radii}"
        assert (
            encoder_channels[-1] == decoder_channels[0]
        ), f"Last encoder channel must match first decoder channel, got last encoder channel {encoder_channels[-1]} != first decoder channel {decoder_channels[0]}"

        self.up_convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for level in range(self.num_levels):
            in_channels = decoder_channels[level]
            out_channels = decoder_channels[level + 1]
            enc_channels = encoder_channels[-(level + 2)]
            curr_neighbor_search_args: ContinuousNeighborSearchArgs = neighbor_search_args.clone(
                radius=neighbor_search_radii[level]
            )
            # Up-sampling convolution
            self.up_convs.append(
                PointConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    neighbor_search_args=curr_neighbor_search_args,
                    use_rel_pos=use_rel_pos,
                    use_rel_pos_encode=use_rel_pos_encode,
                    pos_encode_dim=pos_encode_dim,
                    pos_encode_range=pos_encode_range,
                    reductions=reductions,
                    out_point_type="provided",
                    provided_in_channels=enc_channels,
                )
            )

            # Skip connection
            if enc_channels != out_channels:
                self.skips.append(
                    PointCollectionLinear(
                        in_channels=enc_channels,
                        out_channels=out_channels,
                        bias=True,
                    )
                )
            else:
                self.skips.append(nn.Identity())

            # Additional up-convolution blocks
            up_block = []
            for _ in range(num_blocks_per_level[level]):
                up_block.append(
                    PointConvBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        neighbor_search_args=curr_neighbor_search_args,
                        channel_multiplier=channel_multiplier,
                        use_rel_pos=use_rel_pos,
                        use_rel_pos_encode=use_rel_pos_encode,
                        pos_encode_dim=pos_encode_dim,
                        pos_encode_range=pos_encode_range,
                        reductions=reductions,
                        out_point_feature_type="same",
                    )
                )
            self.up_blocks.append(nn.Sequential(*up_block))

    def forward(
        self, in_point_features: PointCollection, encoder_outs: List[PointCollection]
    ) -> List[PointCollection]:
        out_pcs = []
        out_point_features = in_point_features
        for i, (up_conv, skip, up_block) in enumerate(
            zip(self.up_convs, self.skips, self.up_blocks)
        ):
            out_point_features = up_conv(out_point_features, encoder_outs[-(i + 2)])
            out_point_features = out_point_features + skip(encoder_outs[-(i + 2)])
            out_point_features = up_block(out_point_features)
            out_pcs.append(out_point_features)
        return out_pcs


class PointConvUNetBlock(BaseModule):
    """
    Given an input module, the UNet block will return a list of point collections at each level of the UNet from the inner module.

    +------------+   +------------+   +-------------+   +------------+   +-----------+   +----------------+
    | Down Blocks| ->| Down Conv  | ->| Inner Module| ->| Up Conv    | ->| Up Blocks | ->| Skip Connection|
    +------------+   +------------+   +-------------+   +------------+   +-----------+   +----------------+
    """

    def __init__(
        self,
        in_channels: int,
        inner_module_in_channels: int,
        inner_module_out_channels: int,
        out_channels: int,
        neighbor_search_args: ContinuousNeighborSearchArgs,
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
        reductions: List[REDUCTION_TYPES_STR] = ("mean",),
    ):
        assert inner_module is None or isinstance(inner_module, PointConvUNetBlock)

        super().__init__()
        down_conv_block = [
            PointConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                neighbor_search_args=neighbor_search_args,
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
            out_point_type="downsample",
        )

        self.inner_module = inner_module

        self.up_conv = PointConv(
            in_channels=inner_module_out_channels,
            out_channels=out_channels,
            neighbor_search_args=neighbor_search_args,
            edge_transform_mlp=edge_transform_mlp,
            out_transform_mlp=out_transform_mlp,
            hidden_dim=hidden_dim,
            channel_multiplier=channel_multiplier,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            pos_encode_range=pos_encode_range,
            reductions=reductions,
            out_point_type="provided",
            provided_in_channels=in_channels,
        )

        if num_up_blocks > 0:
            self.up_conv_block = nn.Sequential(
                *[
                    PointConvBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        neighbor_search_args=neighbor_search_args,
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
            self.skip = PointCollectionLinear(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=True,
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
