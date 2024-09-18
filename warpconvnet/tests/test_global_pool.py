import unittest

import torch
import warp as wp

from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.nn.functional.global_pool import global_pool
from warpconvnet.nn.functional.point_pool import REDUCTIONS


class TestGlobalPool(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = torch.device("cuda:0")
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features).to(self.device)
        self.voxel_size = 0.01
        self.st: SpatiallySparseTensor = self.pc.to_sparse(self.voxel_size)

    def test_global_pool_pc(self):
        pooled_pc = global_pool(self.pc, reduce="max")
        self.assertEqual(pooled_pc.batch_size, self.B)
        self.assertEqual(pooled_pc.features.shape[0], self.B)
        self.assertEqual(pooled_pc.features.shape[1], self.C)

    def test_global_pool_st(self):
        pooled_st = global_pool(self.st, reduce="max")
        self.assertEqual(pooled_st.batch_size, self.B)
        self.assertEqual(pooled_st.features.shape[0], self.B)
        self.assertEqual(pooled_st.features.shape[1], self.C)
