import unittest

import torch

import warp as wp
from warp.convnet.geometry.ops.neighbor_search_continuous import (
    NEIGHBOR_SEARCH_MODE,
    NeighborSearchArgs,
)
from warp.convnet.geometry.ops.point_pool import (
    FEATURE_POOLING_MODE,
    FeaturePoolingArgs,
)
from warp.convnet.geometry.point_collection import PointCollection
from warp.convnet.models.point_conv_unet import PointConvUNet


class TestPointConvUNet(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = torch.device("cuda:0")
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features).to(self.device)

    def test_point_conv_unet(self):
        pc = self.pc
        # Create conv layer
        in_channels, out_channels = self.C, 16
        search_args = NeighborSearchArgs(
            mode=NEIGHBOR_SEARCH_MODE.RADIUS,
        )
        pool_args = FeaturePoolingArgs(
            pooling_mode=FEATURE_POOLING_MODE.REDUCTIONS,
            reductions=["mean"],
        )
        down_channels = [16, 32, 64]
        up_channels = [16, 32, 64]
        neighbor_search_radii = [0.1, 0.2]
        downsample_voxel_sizes = [0.1, 0.2]
        conv = PointConvUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            down_channels=down_channels,
            up_channels=up_channels,
            neighbor_search_args=search_args,
            neighbor_search_radii=neighbor_search_radii,
            pooling_args=pool_args,
            downsample_voxel_sizes=downsample_voxel_sizes,
            num_levels=2,
        ).to(self.device)
        # Forward pass
        out = conv(pc)
        # backward
        out[0].feature_tensor.mean().backward()
        # print the conv param grads
        for name, param in conv.named_parameters():
            if param.grad is not None:
                print(name, param.grad.shape)
            else:
                print(name, "has no grad")


if __name__ == "__main__":
    wp.init()
    unittest.main()
