import unittest

import torch
import warp as wp

from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.nn.functional.point_pool import (
    FEATURE_POOLING_MODE,
    REDUCTIONS,
    FeaturePoolingArgs,
    point_collection_pool,
)
from warpconvnet.nn.functional.point_unpool import (
    FEATURE_UNPOOLING_MODE,
    point_collection_unpool,
)


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
        pooled_pc, neighbor_search_result = point_collection_pool(pc, pooling_args)
        self.assertTrue(pooled_pc.batched_features.batch_size == self.B)
        self.assertTrue(pooled_pc.batched_features.batched_tensor.shape[1] == 2 * self.C)

        # Unpool features
        unpooling_mode = FEATURE_UNPOOLING_MODE.REPEAT
        unpooled_pc = point_collection_unpool(
            pooled_pc,
            pc,
            neighbor_search_result,
            concat_unpooled_pc=False,
            unpooling_mode=unpooling_mode,
        )

        # Check if the unpooled features have the same shape
        N_tot = sum(self.Ns)
        self.assertTrue(unpooled_pc.feature_shape == (N_tot, 2 * self.C))

        unpooled_pc = point_collection_unpool(
            pooled_pc,
            pc,
            neighbor_search_result,
            concat_unpooled_pc=True,
            unpooling_mode=unpooling_mode,
        )
        self.assertTrue(unpooled_pc.feature_shape == (N_tot, 3 * self.C))

    def test_point_collection_to_sparse(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        # Create pooling args
        pooling_args = FeaturePoolingArgs(
            pooling_mode=FEATURE_POOLING_MODE.REDUCTIONS,
            reductions=[REDUCTIONS.MEAN, REDUCTIONS.MAX],
            downsample_voxel_size=0.1,
        )
        # Pool features
        pooled_pc, neighbor_search_result = point_collection_pool(pc, pooling_args)
        self.assertTrue(pooled_pc.batched_features.batch_size == self.B)
        self.assertTrue(pooled_pc.batched_features.batched_tensor.shape[1] == 2 * self.C)

        # Convert to sparse tensor
        sparse_tensor = pc.to_sparse(pooling_args.downsample_voxel_size, pooling_args)
        self.assertTrue(sparse_tensor.batched_features.batch_size == self.B)
        self.assertTrue(sparse_tensor.batched_features.batched_tensor.shape[1] == 2 * self.C)
        # Pool features


if __name__ == "__main__":
    wp.init()
    unittest.main()
