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
from warp.convnet.nn.point_conv import PointConv


class TestPointConv(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features).to(self.device)

    def test_point_conv_radius(self):
        pc = self.pc
        # Create conv layer
        in_channels, out_channels = self.C, 16
        search_arg = NeighborSearchArgs(
            mode=NEIGHBOR_SEARCH_MODE.RADIUS,
            radius=0.1,
        )
        conv = PointConv(
            in_channels,
            out_channels,
            neighbor_search_args=search_arg,
        ).to(self.device)
        # Forward pass
        out = conv(pc)
        out.features.mean().backward()
        # print the conv param grads
        for name, param in conv.named_parameters():
            if param.grad is not None:
                print(name, param.grad.shape)
            else:
                print(name, "has no grad")

    def test_point_conv_knn(self):
        pc = self.pc
        # Create conv layer
        in_channels, out_channels = self.C, 16
        search_args = NeighborSearchArgs(
            mode=NEIGHBOR_SEARCH_MODE.KNN,
            k=16,
        )
        conv = PointConv(
            in_channels,
            out_channels,
            neighbor_search_args=search_args,
        ).to(self.device)
        # Forward pass
        out = conv(pc)  # noqa: F841

    def test_point_conv_downsample(self):
        pc = self.pc
        # Create conv layer
        in_channels, out_channels = self.C, 16
        search_args = NeighborSearchArgs(
            mode=NEIGHBOR_SEARCH_MODE.RADIUS,
            radius=0.1,
        )
        pool_args = FeaturePoolingArgs(
            pooling_mode=FEATURE_POOLING_MODE.REDUCTIONS,
            reductions=["mean"],
        )
        conv = PointConv(
            in_channels,
            out_channels,
            neighbor_search_args=search_args,
            pooling_args=pool_args,
            out_point_feature_type="downsample",
            downsample_voxel_size=0.1,
        ).to(self.device)
        # Forward pass
        out = conv(pc)  # noqa: F841


if __name__ == "__main__":
    wp.init()
    unittest.main()
