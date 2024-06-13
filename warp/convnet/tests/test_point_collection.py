import unittest

import torch
import warp as wp
from warp.convnet.geometry.point_collection import PointCollection


class TestPointCollection(unittest.TestCase):

    def setUp(self):
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features)

    # Test point collection construction
    def test_point_collection_construction(self):
        Ns_cumsum = torch.tensor(self.Ns).cumsum(dim=0).tolist()
        self.assertTrue(self.pc.batched_coordinates.batch_size == self.B)
        self.assertTrue(self.pc.batched_coordinates.offsets == [0] + Ns_cumsum)
        self.assertTrue(self.pc.batched_coordinates.batched_tensors.shape == (Ns_cumsum[-1], 3))
        self.assertTrue(self.pc.batched_features.batch_size == self.B)
        self.assertTrue(self.pc.batched_features.batched_tensors.shape == (Ns_cumsum[-1], self.C))

        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        self.assertTrue(pc.batched_coordinates.batched_tensors.device == device)

    # Test point collection sorting
    def test_point_collection_sorting(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        sorted_pc = pc.sort()
        self.assertTrue(sorted_pc.batched_coordinates.batched_tensors.shape == (sum(self.Ns), 3))
        self.assertTrue(sorted_pc.batched_features.batched_tensors.shape == (sum(self.Ns), self.C))

    # Test point collection radius search
    def test_point_collection_radius_search(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        query = torch.rand((self.B * 10000, 3))
        offset = torch.tensor([0] + self.Ns.tolist())
        radius = 0.1
        # indices, distances = pc.radius_search(query, radius)
        # self.assertTrue(distances.shape == (self.B, max(self.Ns), 8))


if __name__ == "__main__":
    unittest.main()
