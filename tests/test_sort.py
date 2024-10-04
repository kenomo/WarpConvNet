import unittest

import torch
import warp as wp

from warpconvnet.core.serialization import POINT_ORDERING, morton_code
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor


class TestSorting(unittest.TestCase):
    def setUp(self):
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features)

        self.voxel_size = 0.01
        self.st_coords = [torch.floor(coords / self.voxel_size).int() for coords in self.coords]
        self.st = SpatiallySparseTensor(self.st_coords, self.features)

    # Test point collection sorting
    def test_point_collection_sorting(self):
        device = torch.device("cuda:0")
        st = self.st.to(device)
        # Get the coordinates and test sorting_permutation
        code, permutation = morton_code(
            coords=st.coordinate_tensor, offsets=st.offsets, ordering=POINT_ORDERING.Z_ORDER
        )
        # min max of the permutation between each offsets should not exceed offset boundaries
        max_code = 0
        for i in range(len(st.offsets) - 1):
            offset = st.offsets[i]
            next_offset = st.offsets[i + 1]
            curr_perm = permutation[offset:next_offset]
            curr_max_code = code[offset:next_offset].max()
            self.assertTrue(curr_max_code >= max_code)
            self.assertTrue(curr_perm.min() >= offset)
            self.assertTrue(curr_perm.max() < next_offset)
            max_code = curr_max_code


if __name__ == "__main__":
    wp.init()
    unittest.main()
