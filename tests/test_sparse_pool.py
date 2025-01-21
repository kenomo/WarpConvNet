import unittest

import torch
import warp as wp

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_coords_ops import generate_output_coords
from warpconvnet.nn.functional.sparse_pool import sparse_reduce, sparse_unpool
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates


class TestSparsePool(unittest.TestCase):
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
        self.st = Voxels(self.coords, self.features, device=device).unique()
        return super().setUp()

    def test_generate_output_coords(self):
        batch_indexed_coords = batch_indexed_coordinates(
            self.st.coordinate_tensor,
            self.st.offsets,
        )
        output_coords, offsets = generate_output_coords(batch_indexed_coords, stride=(2, 2, 2))
        self.assertTrue(output_coords.shape[0] < batch_indexed_coords.shape[0])
        self.assertTrue(offsets.shape == (self.B + 1,))

        st_downsampled = sparse_reduce(self.st, (2, 2, 2), (2, 2, 2), reduction="max")
        self.assertTrue(
            st_downsampled.coordinate_tensor.shape[0] < self.st.coordinate_tensor.shape[0]
        )

        st_downsampled_first = sparse_reduce(self.st, (2, 2, 2), (2, 2, 2), reduction="random")

        self.assertTrue(
            st_downsampled_first.coordinate_tensor.shape[0] < self.st.coordinate_tensor.shape[0]
        )
        self.assertTrue(
            st_downsampled_first.coordinate_tensor.shape[0]
            == st_downsampled.coordinate_tensor.shape[0]
        )

    def test_sparse_unpool(self):
        st_downsampled = sparse_reduce(self.st, (2, 2, 2), (2, 2, 2), reduction="max")
        st_unpooled = sparse_unpool(
            st_downsampled, self.st, (2, 2, 2), (2, 2, 2), concat_unpooled_st=True
        )
        self.assertTrue(
            st_unpooled.coordinate_tensor.shape[0] == self.st.coordinate_tensor.shape[0]
        )


if __name__ == "__main__":
    unittest.main()
