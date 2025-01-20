import unittest

import torch
import warp as wp

from warpconvnet.geometry.coords.search.search_results import RealSearchResult
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig
from warpconvnet.geometry.ops.neighbor_search_continuous import (
    RealSearchMode,
)
from warpconvnet.geometry.types.points import Points


class TestPoints(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = Points(self.coords, self.features)


if __name__ == "__main__":
    wp.init()
    unittest.main()
