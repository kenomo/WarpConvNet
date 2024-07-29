import unittest

import torch

import warp as wp
from warp.convnet.geometry.ops.point_pool import (
    FEATURE_POOLING_MODE,
    REDUCTIONS,
    FeaturePoolingArgs,
    point_collection_pool,
)
from warp.convnet.geometry.point_collection import PointCollection


class TestPointPool(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features)

    # Test point collection construction
    def test_point_pool(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        # Create pooling args
        pooling_args = FeaturePoolingArgs(
            pooling_mode=FEATURE_POOLING_MODE.REDUCTIONS,
            reductions=[REDUCTIONS.MEAN, REDUCTIONS.MAX],
            downsample_voxel_size=0.1,
        )
        # Pool features
        pooled_pc = point_collection_pool(pc, pooling_args)
        self.assertTrue(pooled_pc.batched_features.batch_size == self.B)
        self.assertTrue(pooled_pc.batched_features.batched_tensor.shape[1] == 2 * self.C)


if __name__ == "__main__":
    wp.init()
    unittest.main()
