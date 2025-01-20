import unittest

import torch
import warp as wp

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.normalizations import LayerNorm, RMSNorm


class TestNormalization(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = torch.device("cuda:0")
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)).requires_grad_() for N in self.Ns]
        self.pc = Points(self.coords, self.features).to(self.device)
        self.voxel_size = 0.01
        self.st: Voxels = self.pc.to_sparse(self.voxel_size)

    def test_rms_norm(self):
        rms_norm = RMSNorm(self.C).to(self.device)
        normed_pc = rms_norm(self.pc)
        self.assertEqual(normed_pc.batch_size, self.B)
        self.assertEqual(normed_pc.batched_features.shape[1], self.C)

        # Test the gradient
        normed_pc.features.sum().backward()
        self.assertIsNotNone(rms_norm.weight.grad)

    def test_layer_norm(self):
        layer_norm = LayerNorm(self.C).to(self.device)
        normed_pc = layer_norm(self.pc)

        # Test the gradient
        normed_pc.features.sum().backward()
        self.assertIsNotNone(layer_norm.norm.weight.grad)


if __name__ == "__main__":
    unittest.main()
