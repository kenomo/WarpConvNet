import unittest

import torch
import warp as wp

from warpconvnet.geometry.ops.neighbor_search_continuous import (
    CONTINUOUS_NEIGHBOR_SEARCH_MODE,
    ContinuousNeighborSearchArgs,
)
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.nn.functional.point_pool import (
    FEATURE_POOLING_MODE,
    FeaturePoolingArgs,
)
from warpconvnet.nn.point_conv import PointConv


class TestPointConv(unittest.TestCase):
    def setUp(self):
        wp.init()
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
        search_arg = ContinuousNeighborSearchArgs(
            mode=CONTINUOUS_NEIGHBOR_SEARCH_MODE.RADIUS,
            radius=0.1,
        )
        conv = PointConv(
            in_channels,
            out_channels,
            neighbor_search_args=search_arg,
        ).to(self.device)
        # Forward pass
        out = conv(pc)
        out.feature_tensor.mean().backward()
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
        search_args = ContinuousNeighborSearchArgs(
            mode=CONTINUOUS_NEIGHBOR_SEARCH_MODE.KNN,
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
        search_args = ContinuousNeighborSearchArgs(
            mode=CONTINUOUS_NEIGHBOR_SEARCH_MODE.RADIUS,
            radius=0.1,
        )
        pool_args = FeaturePoolingArgs(
            pooling_mode=FEATURE_POOLING_MODE.REDUCTIONS,
            reductions=["mean"],
            downsample_voxel_size=0.1,
        )
        conv = PointConv(
            in_channels,
            out_channels,
            neighbor_search_args=search_args,
            pooling_args=pool_args,
            out_point_type="downsample",
        ).to(self.device)
        # Forward pass
        out = conv(pc)  # noqa: F841
        assert out.voxel_size is not None
        out.feature_tensor.mean().backward()
        # assert conv params have grad
        for _, param in conv.named_parameters():
            if param.numel() > 0:
                self.assertTrue(param.grad is not None)


if __name__ == "__main__":
    unittest.main()
