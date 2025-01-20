import unittest

import torch
import warp as wp

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.global_pool import global_pool


class TestGlobalPool(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = torch.device("cuda:0")
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = Points(self.coords, self.features).to(self.device)
        self.voxel_size = 0.01
        self.st: Voxels = self.pc.to_sparse(self.voxel_size)

    def test_global_pool_pc(self):
        pooled_pc = global_pool(self.pc, reduce="max")
        self.assertEqual(pooled_pc.batch_size, self.B)
        self.assertEqual(pooled_pc.batched_features.shape[0], self.B)
        self.assertEqual(pooled_pc.batched_features.shape[1], self.C)

    def test_global_pool_st(self):
        pooled_st = global_pool(self.st, reduce="max")
        self.assertEqual(pooled_st.batch_size, self.B)
        self.assertEqual(pooled_st.batched_features.shape[0], self.B)
        self.assertEqual(pooled_st.batched_features.shape[1], self.C)
