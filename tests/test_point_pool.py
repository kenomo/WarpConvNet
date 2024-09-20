import unittest

import torch
import warp as wp

from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.nn.functional.point_pool import REDUCTIONS, point_pool
from warpconvnet.nn.functional.point_unpool import FEATURE_UNPOOLING_MODE, point_unpool


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
        # Pool features
        pooled_pc, neighbor_search_result = point_pool(
            pc,
            reduction=REDUCTIONS.MEAN,
            downsample_voxel_size=0.1,
            return_type="point",
            return_neighbor_search_result=True,
        )
        self.assertTrue(pooled_pc.batch_size == self.B)
        self.assertTrue(pooled_pc.features.shape[1] == self.C)
        # Assert pooled coords are smaller
        self.assertGreater(pc.coordinates.shape[0], pooled_pc.coordinates.shape[0])

        # Unpool features
        unpooling_mode = FEATURE_UNPOOLING_MODE.REPEAT
        unpooled_pc = point_unpool(
            pooled_pc,
            pc,
            concat_unpooled_pc=False,
            unpooling_mode=unpooling_mode,
            pooling_neighbor_search_result=neighbor_search_result,
        )

        # Check if the unpooled features have the same shape
        N_tot = sum(self.Ns)
        self.assertTrue(unpooled_pc.feature_shape == (N_tot, self.C))

        unpooled_pc = point_unpool(
            pooled_pc,
            pc,
            concat_unpooled_pc=True,
            unpooling_mode=unpooling_mode,
            pooling_neighbor_search_result=neighbor_search_result,
        )
        self.assertTrue(unpooled_pc.feature_shape == (N_tot, 2 * self.C))

    def test_point_collection_to_sparse(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        # Pool features
        st = point_pool(pc, reduction=REDUCTIONS.MEAN, downsample_voxel_size=0.1, return_type="sparse")
        self.assertTrue(isinstance(st, SpatiallySparseTensor))
        self.assertTrue(st.batch_size == self.B)
        self.assertTrue(st.features.shape[1] == self.C)

    def test_point_pool_num_points(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        # Pool features
        pooled_pc, neighbor_search_result = point_pool(
            pc,
            reduction=REDUCTIONS.MEAN,
            downsample_num_points=1000,
            return_type="point",
            return_neighbor_search_result=True,
        )
        self.assertTrue(pooled_pc.coordinates.shape[0] == 1000 * self.B)


if __name__ == "__main__":
    wp.init()
    unittest.main()
