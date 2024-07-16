from typing import List, Optional

import torch.nn as nn

from warp.convnet.geometry.ops.neighbor_search_continuous import NeighborSearchArgs
from warp.convnet.geometry.ops.point_pool import FeaturePoolingArgs
from warp.convnet.geometry.point_collection import PointCollection
from warp.convnet.nn.base_module import BaseModel
from warp.convnet.nn.point_conv import PointConvUNetBlock
from warp.convnet.nn.point_transform import PointCollectionTransform
from warp.convnet.ops.reductions import REDUCTION_TYPES

__all__ = ["PointConvUNet"]


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
