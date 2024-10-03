import unittest

import torch
import torch.nn as nn
import warp as wp

from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.nn.functional.point_pool import REDUCTIONS, point_pool
from warpconvnet.nn.functional.point_unpool import FEATURE_UNPOOLING_MODE, point_unpool
from warpconvnet.nn.point_pool import PointMaxPool
from warpconvnet.nn.pools import PointToSparseWrapper
from warpconvnet.utils.ravel import ravel_multi_index_auto_shape


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
        pooled_pc, to_unique = point_pool(
            pc,
            reduction=REDUCTIONS.MEAN,
            downsample_voxel_size=0.1,
            return_type="point",
            return_to_unique=True,
        )
        self.assertTrue(pooled_pc.batch_size == self.B)
        self.assertTrue(pooled_pc.batched_features.shape[1] == self.C)
        # Assert pooled coords are smaller
        self.assertGreater(pc.coordinates.shape[0], pooled_pc.coordinates.shape[0])

        # Unpool features
        unpooling_mode = FEATURE_UNPOOLING_MODE.REPEAT
        unpooled_pc = point_unpool(
            pooled_pc,
            pc,
            concat_unpooled_pc=False,
            unpooling_mode=unpooling_mode,
            to_unique=to_unique,
        )

        # Check if the unpooled features have the same shape
        N_tot = sum(self.Ns)
        self.assertTrue(unpooled_pc.feature_tensor.shape == (N_tot, self.C))

        unpooled_pc = point_unpool(
            pooled_pc,
            pc,
            concat_unpooled_pc=True,
            unpooling_mode=unpooling_mode,
            to_unique=to_unique,
        )
        self.assertTrue(unpooled_pc.feature_tensor.shape == (N_tot, 2 * self.C))

    def test_point_collection_to_sparse(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        # Pool features
        st = point_pool(
            pc, reduction=REDUCTIONS.MEAN, downsample_voxel_size=0.1, return_type="sparse"
        )
        self.assertTrue(isinstance(st, SpatiallySparseTensor))
        self.assertTrue(st.batch_size == self.B)
        self.assertTrue(st.batched_features.shape[1] == self.C)

    def test_point_pool_num_points(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        # Pool features
        pooled_pc = point_pool(
            pc,
            reduction=REDUCTIONS.MEAN,
            downsample_max_num_points=1000,
            return_type="point",
        )
        self.assertGreaterEqual(1000 * self.B, pooled_pc.coordinates.shape[0])

    def test_point_max_pool(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        pooled_pc = point_pool(
            pc,
            reduction=REDUCTIONS.MAX,
            downsample_max_num_points=1000,
            return_type="point",
        )

        self.assertTrue(torch.all(pooled_pc.offsets.diff() <= 1000))
        self.assertTrue(pooled_pc.batched_features.shape[1] == self.C)

    def test_point_to_sparse_wrapper(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)

        voxel_size = 0.1
        raveled_values = ravel_multi_index_auto_shape(torch.floor(pc.batch_indexed_coordinates / voxel_size).int())
        pc = pc.replace(
            batched_features=raveled_values.view(-1, 1)
        )

        wrapper = PointToSparseWrapper(inner_module=nn.Identity(), voxel_size=voxel_size, reduction=REDUCTIONS.MEAN, concat_unpooled_pc=False)
        out_pc = wrapper(pc)
        self.assertTrue(isinstance(out_pc, PointCollection))
        self.assertTrue(out_pc.num_channels == pc.num_channels)
        self.assertTrue(torch.all(out_pc.feature_tensor == pc.feature_tensor))


if __name__ == "__main__":
    unittest.main()
