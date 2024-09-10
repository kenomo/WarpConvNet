import unittest

import torch

import warp as wp
from warp.convnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warp.convnet.nn.functional.sparse_coords_ops import (
    expand_coords,
    generate_output_coords,
)
from warp.convnet.utils.batch_index import batch_indexed_coordinates


class TestSparseOps(unittest.TestCase):
    def setUp(self) -> None:
        wp.init()
        # Set random seed
        torch.manual_seed(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B, min_N, max_N, self.C = 3, 100000, 1000000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.voxel_size = 0.01
        self.coords = [(torch.rand((N, 3)) / self.voxel_size).int() for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.st = SpatiallySparseTensor(self.coords, self.features, device=device).unique()
        return super().setUp()

    def test_generate_output_coords(self):
        batch_indexed_coords = batch_indexed_coordinates(
            self.st.coordinate_tensor,
            self.st.offsets,
        )
        output_coords, offsets = generate_output_coords(batch_indexed_coords, stride=(2, 2, 2))
        self.assertTrue(output_coords.shape[0] < batch_indexed_coords.shape[0])
        self.assertTrue(offsets.shape == (self.B + 1,))

        st_upsampled = expand_coords(self.st, (3, 3, 3), (1, 1, 1))
        self.assertTrue(
            st_upsampled.coordinate_tensor.shape[0] > self.st.coordinate_tensor.shape[0]
        )


if __name__ == "__main__":
    unittest.main()
