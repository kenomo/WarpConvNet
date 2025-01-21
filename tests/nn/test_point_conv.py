import unittest

import torch
import warp as wp

from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig, RealSearchMode
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.point_conv import PointConv
from warpconvnet.ops.reductions import REDUCTIONS


class TestPointConv(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = torch.device("cuda:0")
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = Points(self.coords, self.features).to(self.device)

    def test_point_conv_radius(self):
        pc = self.pc
        # Create conv layer
        in_channels, out_channels = self.C, 16
        search_arg = RealSearchConfig(
            mode=RealSearchMode.RADIUS,
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
        search_args = RealSearchConfig(
            mode=RealSearchMode.KNN,
            knn_k=16,
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
        search_args = RealSearchConfig(
            mode=RealSearchMode.RADIUS,
            radius=0.1,
        )
        conv = PointConv(
            in_channels,
            out_channels,
            neighbor_search_args=search_args,
            pooling_reduction=REDUCTIONS.MEAN,
            pooling_voxel_size=0.1,
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
