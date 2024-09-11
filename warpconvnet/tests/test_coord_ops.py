import unittest

import torch

import warp as wp
from warp.convnet.geometry.ops.coord_ops import relative_coords
from warp.convnet.geometry.ops.neighbor_search_continuous import (
    CONTINUOUS_NEIGHBOR_SEARCH_MODE,
    ContinuousNeighborSearchArgs,
    NeighborSearchResult,
)
from warp.convnet.geometry.point_collection import PointCollection


class TestPointCollection(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features)

    # Test point collection radius search
    def test_relative_coords(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        radius = 0.1
        args = ContinuousNeighborSearchArgs(
            mode=CONTINUOUS_NEIGHBOR_SEARCH_MODE.RADIUS,
            radius=radius,
        )
        search_result = pc.batched_coordinates.neighbors(args)
        self.assertTrue(isinstance(search_result, NeighborSearchResult))

        # Test relative_coords
        rel_coords = relative_coords(
            pc.batched_coordinates.batched_tensor,
            search_result,
        )
        distance = torch.norm(rel_coords, dim=-1)
        self.assertTrue(rel_coords.shape == (search_result.neighbors_row_splits[-1].item(), 3))
        self.assertTrue((distance <= radius).all().item())


if __name__ == "__main__":
    wp.init()
    unittest.main()
