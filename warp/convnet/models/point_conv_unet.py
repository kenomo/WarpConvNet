from typing import List, Optional

import torch.nn as nn

from warp.convnet.geometry.ops.neighbor_search_continuous import NeighborSearchArgs
from warp.convnet.geometry.ops.point_pool import (
    FeaturePoolingArgs,
    point_collection_pool,
)
from warp.convnet.geometry.ops.point_unpool import point_collection_unpool
from warp.convnet.geometry.point_collection import PointCollection
from warp.convnet.nn.base_module import BaseModel
from warp.convnet.nn.point_conv import PointConvUNetBlock
from warp.convnet.nn.point_transform import PointCollectionTransform
from warp.convnet.ops.reductions import REDUCTION_TYPES_STR

__all__ = ["PointConvUNet"]


class PointConvUNet(BaseModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_channels: List[int],
        up_channels: List[int],
        neighbor_search_args: NeighborSearchArgs,
        neighbor_search_radii: List[float],
        pooling_args: FeaturePoolingArgs,
        downsample_voxel_sizes: List[float],
        initial_downsample_voxel_size: Optional[float] = None,
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
        num_levels: int = 4,
    ):
        super().__init__()
        assert len(down_channels) == num_levels + 1 and len(up_channels) == num_levels + 1
        assert len(downsample_voxel_sizes) == num_levels
        assert len(neighbor_search_radii) == num_levels
        for i in range(num_levels - 1):
            assert downsample_voxel_sizes[i] < downsample_voxel_sizes[i + 1]
            assert neighbor_search_radii[i] < neighbor_search_radii[i + 1]

        self.num_levels = num_levels
        self.in_map = PointCollectionTransform(nn.Linear(in_channels, down_channels[0]))

        # Create from the deepest level to the shallowest level
        inner_block = None
        # start from the innermost block. This has the largest receptive field, radius, and voxel size
        for i in range(num_levels - 1, -1, -1):
            curr_neighbor_search_args: NeighborSearchArgs = neighbor_search_args.clone(
                radius=neighbor_search_radii[i]
            )
            curr_pooling_args: FeaturePoolingArgs = pooling_args.clone(
                downsample_voxel_size=downsample_voxel_sizes[i]
            )
            inner_block = PointConvUNetBlock(
                inner_module=inner_block,
                in_channels=down_channels[i],
                inner_module_in_channels=down_channels[i + 1],
                inner_module_out_channels=up_channels[i + 1],
                out_channels=up_channels[i],
                neighbor_search_args=curr_neighbor_search_args,
                pooling_args=curr_pooling_args,
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
            )
        self.unet = inner_block
        if initial_downsample_voxel_size is None:
            initial_downsample_voxel_size = downsample_voxel_sizes[0] / 2
        self.initial_pooling_args = pooling_args.clone(
            downsample_voxel_size=initial_downsample_voxel_size
        )

        self.out_map = PointCollectionTransform(
            nn.Linear(up_channels[0] + in_channels, out_channels)
        )

    def forward(self, in_pc: PointCollection) -> List[PointCollection]:
        """
        Given an input point collection, the network will return a list of point collections at each level of the UNet.
        """

        # downsample
        pooled_pc, nsearch = point_collection_pool(in_pc, self.initial_pooling_args)
        out = self.in_map(pooled_pc)

        # forward pass through the UNet
        out = self.unet(out)

        # upsample
        unpooled_pc = point_collection_unpool(out[0], in_pc, nsearch, concat_unpooled_pc=True)
        unpooled_pc = self.out_map(unpooled_pc)
        return unpooled_pc, *out
